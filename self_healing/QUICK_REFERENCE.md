# Self-Healing System - Quick Reference

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Basic Usage (Backward Compatible)

```python
from self_healing.health_monitor import HealthMonitor
from self_healing.drift_detector import DriftDetector
from self_healing.decision_engine import DecisionEngine

# Old code still works!
monitor = HealthMonitor(recall_threshold=0.85)
detector = DriftDetector()
engine = DecisionEngine()

health = monitor.evaluate(y_true, y_pred)
drift = detector.detect(X_ref, X_cur)
decision = engine.decide(health, drift)
```

### 2. Enhanced Workflow (Recommended)

```python
from self_healing import SelfHealingOrchestrator

# Initialize orchestrator
orchestrator = SelfHealingOrchestrator()

# Run complete workflow
report = orchestrator.run_healing_workflow(
    X_reference=X_clean,
    X_current=X_production,
    y_current=y_production,
    current_model=model,
    X_train=X_train,
    y_train=y_train
)

# Check results
print(f"Success: {report['success']}")
print(f"Action: {report['final_action']}")
```

## Configuration

### Option 1: Default Config

```python
from self_healing import SelfHealingConfig

config = SelfHealingConfig()  # Uses all defaults
```

### Option 2: YAML File

```python
from self_healing import load_config

config = load_config("config/self_healing.yaml")
```

### Option 3: Programmatic

```python
from self_healing import SelfHealingConfig, DecisionEngineConfig

config = SelfHealingConfig()
config.decision_engine.policy = "aggressive"
config.decision_engine.max_retrains_per_day = 5
```

## Key Components

### Health Monitor

```python
from self_healing import HealthMonitor, HealthMonitorConfig

config = HealthMonitorConfig(
    recall_threshold=0.85,
    enable_trend_tracking=True
)
monitor = HealthMonitor(config)

result = monitor.evaluate(y_true, y_pred, y_pred_proba)
# Returns: is_healthy, metrics, degradation, confusion_matrix
```

### Drift Detector

```python
from self_healing import DriftDetector, DriftDetectorConfig

config = DriftDetectorConfig(
    methods=["ks", "psi", "wasserstein"],
    enable_ensemble=True
)
detector = DriftDetector(config)

result = detector.detect(X_reference, X_current)
# Returns: drift_detected, drift_ratio, drifted_features
```

### Decision Engine

```python
from self_healing import DecisionEngine, DecisionEngineConfig

config = DecisionEngineConfig(
    policy="balanced",  # or "conservative", "aggressive"
    max_retrains_per_day=3
)
engine = DecisionEngine(config)

decision = engine.decide(health_status, drift_status)
# Returns: action, confidence, reasoning
```

### Retrainer

```python
from self_healing import Retrainer, RetrainerConfig

config = RetrainerConfig(
    model_type="random_forest",
    enable_hyperparameter_tuning=True
)
retrainer = Retrainer(config)

new_model = retrainer.retrain(X_train, y_train)
metadata = retrainer.save(new_model, "models/new_model.joblib")
```

### Rollback Manager

```python
from self_healing import RollbackManager, RollbackConfig

manager = RollbackManager(models_dir="models/baseline")

# Save version
version_id = manager.save_version(model, version_tag="stable")

# Rollback
model = manager.rollback(version_tag="stable")

# List versions
versions = manager.list_versions()
```

### Alert System

```python
from self_healing import AlertSystem, AlertSystemConfig

config = AlertSystemConfig(
    enable_slack=True,
    slack_webhook_url="YOUR_WEBHOOK"
)
alerts = AlertSystem(config)

alerts.send_alert(
    title="Model Drift",
    message="Drift detected",
    severity="WARNING"
)
```

## Decision Policies

### Conservative
- Only acts when critical
- Prefers investigation over automatic fixes
- Best for: High-stakes applications

### Balanced (Default)
- Moderate approach
- Automatic fixing for clear issues
- Manual review for edge cases
- Best for: Most production systems

### Aggressive
- Proactive retraining
- Act on any degradation signal
- Best for: Fast-moving domains

## Action Types

| Action | Description | Trigger |
|--------|-------------|---------|
| `no_action` | Everything normal | Health good, no drift |
| `monitor` | Watch closely | Minor drift detected |
| `alert` | Notify team | Warning levels reached |
| `investigate` | Manual review needed | Unusual patterns |
| `retrain` | Automatic retraining | Critical + drift |
| `rollback` | Restore previous | Critical without drift |

## Environment Variables

```bash
# Override any config via env vars
export SELFHEALING_DECISION_ENGINE_POLICY=aggressive
export SELFHEALING_HEALTH_MONITOR_RECALL_THRESHOLD=0.90
export SELFHEALING_ALERT_SYSTEM_SLACK_WEBHOOK_URL=https://...
```

## Logging

```python
import logging

# Configure logging level
logging.basicConfig(level=logging.INFO)

# Component loggers
logger = logging.getLogger('self_healing.orchestrator')
logger.setLevel(logging.DEBUG)
```

## Common Patterns

### Daily Scheduled Healing

```python
import schedule
from self_healing import SelfHealingOrchestrator

orchestrator = SelfHealingOrchestrator()

def daily_healing():
    # Load fresh data
    X_ref, X_cur, y_cur = load_data()
    model = load_model()
    
    report = orchestrator.run_healing_workflow(
        X_reference=X_ref,
        X_current=X_cur,
        y_current=y_cur,
        current_model=model
    )
    
    log_report(report)

schedule.every().day.at("02:00").do(daily_healing)
```

### Dry-Run Testing

```python
# Test without making changes
report = orchestrator.run_healing_workflow(
    ...,
    dry_run=True  # No actual retraining/rollback
)

print(f"Would perform: {report['final_action']}")
```

### A/B Testing

```python
# Test new policy without committing
config_test = SelfHealingConfig()
config_test.decision_engine.policy = "aggressive"

orchestrator_test = SelfHealingOrchestrator(config_test)
report_test = orchestrator_test.run_healing_workflow(..., dry_run=True)

# Compare with current policy
# If better, switch to aggressive
```

## Troubleshooting

### Import Errors

```bash
# Verify installation
python -c "from self_healing import SelfHealingOrchestrator; print('OK')"

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### No Alerts Received

```python
# Check alert system configuration
orchestrator.alert_system.get_alert_summary()

# Test alert manually
orchestrator.alert_system.send_alert(
    "Test", "Testing alerts", "INFO", force=True
)
```

### Rate Limiting Issues

```python
# Check rate limits
summary = orchestrator.decision_engine.get_decision_summary()
print(f"Retrains last 24h: {summary['retrains_last_24h']}")

# Reset rate limits (emergency)
orchestrator.decision_engine.reset_rate_limits()
```

## Performance Tips

1. **Disable Hyperparameter Tuning** in production (expensive)
2. **Use Ensemble Drift Detection** for better accuracy
3. **Enable Adaptive Thresholds** for evolving systems
4. **Set Appropriate Cooldown Periods** to avoid thrashing
5. **Use Incremental Learning** for large datasets

## Files Reference

- `config.py` - Configuration models
- `exceptions.py` - Custom exceptions
- `health_monitor.py` - Health monitoring
- `drift_detector.py` - Drift detection
- `decision_engine.py` - Decision logic
- `retrainer.py` - Model retraining
- `rollback.py` - Version management
- `alert_system.py` - Alerting
- `orchestrator.py` - Workflow coordination
- `__init__.py` - Package exports

## Support

For issues or questions:
1. Check walkthrough.md for detailed documentation
2. Review implementation_plan.md for architecture
3. Run demonstration: `python notebooks/self_healing_demo.py`
4. Check logs for detailed error messages
