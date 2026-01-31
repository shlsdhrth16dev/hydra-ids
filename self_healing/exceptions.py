"""
Custom exceptions for self-healing system.

Provides a hierarchy of exceptions for better error handling
and debugging across all self-healing components.
"""


class SelfHealingException(Exception):
    """Base exception for all self-healing errors."""
    pass


class ConfigurationError(SelfHealingException):
    """Raised when configuration is invalid."""
    pass


class HealthMonitorError(SelfHealingException):
    """Raised when health monitoring fails."""
    pass


class DriftDetectionError(SelfHealingException):
    """Raised when drift detection fails."""
    pass


class DecisionEngineError(SelfHealingException):
    """Raised when decision engine encounters an error."""
    pass


class RetrainingError(SelfHealingException):
    """Raised when retraining fails."""
    pass


class RollbackError(SelfHealingException):
    """Raised when rollback operation fails."""
    pass


class OrchestrationError(SelfHealingException):
    """Raised when orchestration workflow fails."""
    pass


class AlertError(SelfHealingException):
    """Raised when alert system fails."""
    pass


class ValidationError(SelfHealingException):
    """Raised when data or model validation fails."""
    pass


class VersioningError(SelfHealingException):
    """Raised when version management fails."""
    pass
