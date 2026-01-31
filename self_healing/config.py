"""
Configuration management for self-healing system.

Provides Pydantic-based configuration models with validation,
support for YAML/JSON config files, and environment variable overrides.
"""

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator
from pathlib import Path
import yaml
import json
import os


class HealthMonitorConfig(BaseModel):
    """Configuration for health monitoring."""
    
    recall_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    accuracy_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    precision_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    f1_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    
    # Trend analysis
    enable_trend_tracking: bool = True
    trend_window_size: int = Field(default=10, ge=1)
    degradation_threshold: float = Field(default=0.05, ge=0.0)
    
    # Adaptive thresholds
    enable_adaptive_thresholds: bool = True
    adaptation_rate: float = Field(default=0.1, ge=0.0, le=1.0)


class DriftDetectorConfig(BaseModel):
    """Configuration for drift detection."""
    
    # Statistical tests
    ks_p_value_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    psi_threshold: float = Field(default=0.2, ge=0.0)
    wasserstein_threshold: float = Field(default=0.3, ge=0.0)
    
    # Drift ratio
    drift_ratio_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Ensemble voting
    enable_ensemble: bool = True
    min_tests_agree: int = Field(default=2, ge=1)
    
    # Feature weighting
    enable_feature_importance_weighting: bool = True
    
    # Methods to use
    methods: List[Literal["ks", "psi", "wasserstein"]] = ["ks", "psi", "wasserstein"]


class DecisionEngineConfig(BaseModel):
    """Configuration for decision engine."""
    
    # Decision thresholds
    critical_health_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    warning_health_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    
    critical_drift_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    warning_drift_threshold: float = Field(default=0.2, ge=0.0, le=1.0)
    
    # Rate limiting
    enable_rate_limiting: bool = True
    max_retrains_per_day: int = Field(default=3, ge=0)
    cooldown_hours: float = Field(default=6.0, ge=0.0)
    
    # Confidence scoring
    min_confidence_for_action: float = Field(default=0.7, ge=0.0, le=1.0)
    
    # Decision policy
    policy: Literal["conservative", "balanced", "aggressive"] = "balanced"


class RetrainerConfig(BaseModel):
    """Configuration for model retraining."""
    
    # Training mode
    enable_incremental_learning: bool = False
    enable_warm_start: bool = True
    
    # Model settings
    model_type: Literal["random_forest", "xgboost", "sgd"] = "random_forest"
    n_estimators: int = Field(default=200, ge=1)
    max_depth: Optional[int] = Field(default=None, ge=1)
    
    # Hyperparameter optimization
    enable_hyperparameter_tuning: bool = False
    tuning_trials: int = Field(default=50, ge=1)
    tuning_timeout_seconds: int = Field(default=600, ge=1)
    
    # Data validation
    min_training_samples: int = Field(default=1000, ge=1)
    max_class_imbalance_ratio: float = Field(default=100.0, ge=1.0)
    
    # Class imbalance handling
    handle_class_imbalance: bool = True
    imbalance_strategy: Literal["class_weight", "smote", "downsample"] = "class_weight"


class RollbackConfig(BaseModel):
    """Configuration for rollback management."""
    
    # Versioning
    max_versions_to_keep: int = Field(default=10, ge=1)
    auto_cleanup_old_versions: bool = True
    
    # Validation
    validate_after_rollback: bool = True
    validation_sample_size: int = Field(default=1000, ge=1)
    
    # Model registry
    use_mlflow_registry: bool = False
    mlflow_tracking_uri: Optional[str] = None


class AlertSystemConfig(BaseModel):
    """Configuration for alert system."""
    
    # Alert channels
    enable_slack: bool = False
    enable_email: bool = False
    enable_webhook: bool = False
    
    # Slack settings
    slack_webhook_url: Optional[str] = None
    slack_channel: Optional[str] = None
    
    # Email settings
    email_recipients: List[str] = []
    email_from: Optional[str] = None
    smtp_server: Optional[str] = None
    
    # Webhook settings
    webhook_url: Optional[str] = None
    
    # Alert severity
    min_severity_level: Literal["INFO", "WARNING", "CRITICAL"] = "WARNING"
    
    # Deduplication
    enable_deduplication: bool = True
    dedup_window_minutes: int = Field(default=30, ge=1)


class OrchestratorConfig(BaseModel):
    """Configuration for orchestrator."""
    
    # Workflow settings
    enable_async_operations: bool = False
    enable_dry_run: bool = False
    
    # Scheduling
    enable_scheduled_healing: bool = True
    healing_interval_hours: float = Field(default=24.0, ge=0.1)
    
    # Concurrency
    max_concurrent_healings: int = Field(default=1, ge=1)
    
    # Distributed locks
    enable_distributed_locks: bool = False
    lock_timeout_seconds: int = Field(default=300, ge=1)
    
    # Circuit breaker
    enable_circuit_breaker: bool = True
    failure_threshold: int = Field(default=5, ge=1)
    recovery_timeout_seconds: int = Field(default=60, ge=1)


class SelfHealingConfig(BaseModel):
    """Master configuration for self-healing system."""
    
    # Component configs
    health_monitor: HealthMonitorConfig = Field(default_factory=HealthMonitorConfig)
    drift_detector: DriftDetectorConfig = Field(default_factory=DriftDetectorConfig)
    decision_engine: DecisionEngineConfig = Field(default_factory=DecisionEngineConfig)
    retrainer: RetrainerConfig = Field(default_factory=RetrainerConfig)
    rollback: RollbackConfig = Field(default_factory=RollbackConfig)
    alert_system: AlertSystemConfig = Field(default_factory=AlertSystemConfig)
    orchestrator: OrchestratorConfig = Field(default_factory=OrchestratorConfig)
    
    # Global settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_file: Optional[str] = None
    
    # Paths
    models_dir: str = "models/baseline"
    data_dir: str = "data/processed"
    metrics_dir: str = "metrics/self_healing"
    
    @classmethod
    def from_yaml(cls, path: str) -> "SelfHealingConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    @classmethod
    def from_json(cls, path: str) -> "SelfHealingConfig":
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def to_yaml(self, path: str) -> None:
        """Save configuration to YAML file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.safe_dump(self.model_dump(), f, default_flow_style=False)
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def from_env(cls, prefix: str = "SELFHEALING_") -> "SelfHealingConfig":
        """
        Load configuration from environment variables.
        
        Example: SELFHEALING_HEALTH_MONITOR_RECALL_THRESHOLD=0.9
        """
        config = cls()
        
        # Parse environment variables
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Remove prefix and convert to lowercase
                config_key = key[len(prefix):].lower()
                parts = config_key.split('_')
                
                # Navigate nested config
                current = config
                for part in parts[:-1]:
                    if hasattr(current, part):
                        current = getattr(current, part)
                
                # Set value with type conversion
                if hasattr(current, parts[-1]):
                    field_type = type(getattr(current, parts[-1]))
                    try:
                        if field_type == bool:
                            setattr(current, parts[-1], value.lower() in ('true', '1', 'yes'))
                        elif field_type == int:
                            setattr(current, parts[-1], int(value))
                        elif field_type == float:
                            setattr(current, parts[-1], float(value))
                        else:
                            setattr(current, parts[-1], value)
                    except ValueError:
                        pass  # Skip invalid conversions
        
        return config


# Default configuration instance
DEFAULT_CONFIG = SelfHealingConfig()


def load_config(
    config_path: Optional[str] = None,
    config_format: Literal["yaml", "json"] = "yaml",
    use_env: bool = True
) -> SelfHealingConfig:
    """
    Load configuration from file and/or environment variables.
    
    Args:
        config_path: Path to configuration file
        config_format: Format of config file ("yaml" or "json")
        use_env: Whether to override with environment variables
    
    Returns:
        Loaded and validated configuration
    """
    # Start with default config
    if config_path and Path(config_path).exists():
        if config_format == "yaml":
            config = SelfHealingConfig.from_yaml(config_path)
        else:
            config = SelfHealingConfig.from_json(config_path)
    else:
        config = SelfHealingConfig()
    
    # Override with environment variables
    if use_env:
        env_config = SelfHealingConfig.from_env()
        # Merge configurations (env takes precedence)
        # This is a simple implementation; could be more sophisticated
        config = env_config
    
    return config
