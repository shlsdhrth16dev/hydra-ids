"""
Attacks Module - Production-Grade Adversarial Testing Framework

This module provides comprehensive attack simulation capabilities for testing
ML-based Intrusion Detection Systems.

Available Attack Types:
- Poisoning: Label flipping and feature noise injection
- Evasion: Adversarial perturbations to bypass detection
- Drift: Concept/covariate drift simulation
- Corruption: Feature dropping, missing values, outliers

Components:
- attack_controller: Orchestrates attacks with metrics
- attack_simulator: Streams attack traffic from test data
- attack_metrics: Measures attack effectiveness
- attack_validator: Input validation utilities
"""

__version__ = '1.0.0'

from .attack_controller import AttackController
from .attack_simulator import AttackSimulator
from .attack_metrics import AttackMetrics
from .poisoning import label_flipping_attack, feature_noise_attack
from .evasion import evasion_noise, targeted_evasion
from .drift import gradual_mean_shift, covariate_drift
from .corruption import drop_features, inject_missing_values, inject_outliers

__all__ = [
    'AttackController',
    'AttackSimulator',
    'AttackMetrics',
    'label_flipping_attack',
    'feature_noise_attack',
    'evasion_noise',
    'targeted_evasion',
    'gradual_mean_shift',
    'covariate_drift',
    'drop_features',
    'inject_missing_values',
    'inject_outliers',
]
