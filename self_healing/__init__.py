"""
Self-Healing and Recovery System for IDS.

Production-grade self-healing framework with comprehensive monitoring,
drift detection, intelligent decision making, and automated recovery.
"""

from .config import (
    SelfHealingConfig,
    HealthMonitorConfig,
    DriftDetectorConfig,
    DecisionEngineConfig,
    RetrainerConfig,
    RollbackConfig,
    AlertSystemConfig,
    OrchestratorConfig,
    DEFAULT_CONFIG,
    load_config,
)
from .exceptions import (
    SelfHealingException,
    ConfigurationError,
    HealthMonitorError,
    DriftDetectionError,
    DecisionEngineError,
    RetrainingError,
    RollbackError,
    OrchestrationError,
    AlertError,
    ValidationError,
    VersioningError,
)
from .health_monitor import HealthMonitor
from .drift_detector import DriftDetector
from .decision_engine import DecisionEngine
from .retrainer import Retrainer
from .rollback import RollbackManager
from .alert_system import AlertSystem
from .orchestrator import SelfHealingOrchestrator, HealingState

__version__ = "2.0.0"

__all__ = [
    # Main orchestrator
    "SelfHealingOrchestrator",
    "HealingState",
    
    # Core components
    "HealthMonitor",
    "DriftDetector",
    "DecisionEngine",
    "Retrainer",
    "RollbackManager",
    "AlertSystem",
    
    # Configuration
    "SelfHealingConfig",
    "HealthMonitorConfig",
    "DriftDetectorConfig",
    "DecisionEngineConfig",
    "RetrainerConfig",
    "RollbackConfig",
    "AlertSystemConfig",
    "OrchestratorConfig",
    "DEFAULT_CONFIG",
    "load_config",
    
    # Exceptions
    "SelfHealingException",
    "ConfigurationError",
    "HealthMonitorError",
    "DriftDetectionError",
    "DecisionEngineError",
    "RetrainingError",
    "RollbackError",
    "OrchestrationError",
    "AlertError",
    "ValidationError",
    "VersioningError",
]
