"""
Self-Healing Orchestrator - Central Coordinator.

Coordinates all self-healing components in a complete workflow
with state management, error handling, and comprehensive reporting.
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from enum import Enum
import logging
import pandas as pd
import numpy as np

from .config import SelfHealingConfig
from .health_monitor import HealthMonitor
from .drift_detector import DriftDetector
from .decision_engine import DecisionEngine
from .retrainer import Retrainer
from .rollback import RollbackManager
from .alert_system import AlertSystem
from .exceptions import OrchestrationError, SelfHealingException


logger = logging.getLogger(__name__)


class HealingState(Enum):
    """States in the self-healing workflow."""
    IDLE = "idle"
    MONITORING = "monitoring"
    ANALYZING = "analyzing"
    DECIDING = "deciding"
    ACTING = "acting"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


class SelfHealingOrchestrator:
    """
    Central orchestrator for self-healing workflows.
    
    Features:
    - Complete workflow coordination
    - State machine management
    - Error handling and recovery
    - Comprehensive reporting
    - Dry-run mode for testing
    - Alert integration
    
    Args:
        config: SelfHealingConfig instance
    """
    
    def __init__(self, config: Optional[SelfHealingConfig] = None):
        if config is None:
            config = SelfHealingConfig()
        
        self.config = config
        
        # Initialize components
        self.health_monitor = HealthMonitor(config.health_monitor)
        self.drift_detector = DriftDetector(config.drift_detector)
        self.decision_engine = DecisionEngine(config.decision_engine)
        self.retrainer = Retrainer(config.retrainer)
        self.rollback_manager = RollbackManager(config.rollback, config.models_dir)
        self.alert_system = AlertSystem(config.alert_system)
        
        # Workflow state
        self.current_state = HealingState.IDLE
        self.workflow_history: List[Dict[str, Any]] = []
        
        logger.info("SelfHealingOrchestrator initialized with config")
    
    def run_healing_workflow(
        self,
        X_reference: pd.DataFrame,
        X_current: pd.DataFrame,
        y_current: pd.Series,
        current_model: Any,
        X_train: Optional[pd.DataFrame] = None,
        y_train: Optional[pd.Series] = None,
        dry_run: bool = None
    ) -> Dict[str, Any]:
        """
        Execute complete self-healing workflow.
        
        Args:
            X_reference: Reference feature data for drift detection
            X_current: Current production feature data
            y_current: Current true labels
            current_model: Current production model
            X_train: Training data for retraining (optional)
            y_train: Training labels for retraining (optional)
            dry_run: Override dry-run mode
        
        Returns:
            Comprehensive workflow report
        
        Raises:
            OrchestrationError: If workflow fails critically
        """
        if dry_run is None:
            dry_run = self.config.orchestrator.enable_dry_run
        
        workflow_start = datetime.now()
        workflow_id = f"healing_{workflow_start.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("="*80)
        logger.info("Starting self-healing workflow: %s (dry_run=%s)", workflow_id, dry_run)
        logger.info("="*80)
        
        workflow_report = {
            'workflow_id': workflow_id,
            'start_time': workflow_start.isoformat(),
            'dry_run': dry_run,
            'stages': {},
            'final_action': None,
            'success': False,
        }
        
        try:
            # Stage 1: Health Monitoring
            self._transition_state(HealingState.MONITORING)
            health_result = self._stage_health_monitoring(
                current_model, X_current, y_current, workflow_report
            )
            
            # Stage 2: Drift Detection
            self._transition_state(HealingState.ANALYZING)
            drift_result = self._stage_drift_detection(
                X_reference, X_current, workflow_report
            )
            
            # Stage 3: Decision Making
            self._transition_state(HealingState.DECIDING)
            decision_result = self._stage_decision_making(
                health_result, drift_result, workflow_report
            )
            
            # Stage 4: Execute Action
            self._transition_state(HealingState.ACTING)
            action_result = self._stage_execute_action(
                decision_result,
                current_model,
                X_train,
                y_train,
                X_current,
                y_current,
                dry_run,
                workflow_report
            )
            
            # Stage 5: Validation (if applicable)
            if action_result.get('new_model') is not None:
                self._transition_state(HealingState.VALIDATING)
                validation_result = self._stage_validation(
                    action_result['new_model'],
                    X_current,
                    y_current,
                    workflow_report
                )
            
            # Workflow completed successfully
            self._transition_state(HealingState.COMPLETED)
            workflow_report['success'] = True
            workflow_report['final_action'] = decision_result['action']
            
            # Send success alert
            if decision_result['action'] not in ["no_action", "monitor"]:
                self.alert_system.send_alert(
                    title=f"Self-Healing Workflow Completed: {decision_result['action']}",
                    message=f"Workflow {workflow_id} completed successfully",
                    severity="INFO",
                    context={
                        'action': decision_result['action'],
                        'health_status': health_result['is_healthy'],
                        'drift_detected': drift_result['drift_detected']
                    }
                )
            
        except Exception as e:
            logger.error("Workflow failed: %s", str(e), exc_info=True)
            self._transition_state(HealingState.FAILED)
            workflow_report['success'] = False
            workflow_report['error'] = str(e)
            
            # Send failure alert
            self.alert_system.send_alert(
                title=f"Self-Healing Workflow Failed",
                message=f"Workflow {workflow_id} encountered an error: {str(e)}",
                severity="CRITICAL",
                context={'workflow_id': workflow_id, 'error': str(e)}
            )
            
            if not self.config.orchestrator.enable_circuit_breaker:
                raise OrchestrationError(f"Workflow failed: {e}") from e
        
        finally:
            # Finalize report
            workflow_end = datetime.now()
            workflow_report['end_time'] = workflow_end.isoformat()
            workflow_report['duration_seconds'] = (workflow_end - workflow_start).total_seconds()
            workflow_report['final_state'] = self.current_state.value
            
            # Store in history
            self.workflow_history.append(workflow_report)
            
            logger.info("="*80)
            logger.info("Workflow completed: %s (success=%s, duration=%.2fs)",
                       workflow_id, workflow_report['success'],
                       workflow_report['duration_seconds'])
            logger.info("="*80)
        
        return workflow_report
    
    def _stage_health_monitoring(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute health monitoring stage."""
        logger.info("Stage 1: Health Monitoring")
        
        try:
            # Get predictions
            y_pred = model.predict(X)
            
            # Get probabilities if available
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X)
            
            # Evaluate health
            health_result = self.health_monitor.evaluate(y, y_pred, y_pred_proba)
            
            report['stages']['health_monitoring'] = {
                'success': True,
                'result': health_result
            }
            
            logger.info("Health: is_healthy=%s, recall=%.4f, precision=%.4f",
                       health_result['is_healthy'],
                       health_result['metrics']['recall'],
                       health_result['metrics']['precision'])
            
            return health_result
            
        except Exception as e:
            logger.error("Health monitoring stage failed: %s", str(e))
            report['stages']['health_monitoring'] = {'success': False, 'error': str(e)}
            raise
    
    def _stage_drift_detection(
        self,
        X_ref: pd.DataFrame,
        X_cur: pd.DataFrame,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute drift detection stage."""
        logger.info("Stage 2: Drift Detection")
        
        try:
            drift_result = self.drift_detector.detect(X_ref, X_cur)
            
            report['stages']['drift_detection'] = {
                'success': True,
                'result': drift_result
            }
            
            logger.info("Drift: detected=%s, ratio=%.4f, methods=%s",
                       drift_result['drift_detected'],
                       drift_result['drift_ratio'],
                       drift_result['methods_used'])
            
            return drift_result
            
        except Exception as e:
            logger.error("Drift detection stage failed: %s", str(e))
            report['stages']['drift_detection'] = {'success': False, 'error': str(e)}
            raise
    
    def _stage_decision_making(
        self,
        health_result: Dict[str, Any],
        drift_result: Dict[str, Any],
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute decision making stage."""
        logger.info("Stage 3: Decision Making")
        
        try:
            decision_result = self.decision_engine.decide(health_result, drift_result)
            
            report['stages']['decision_making'] = {
                'success': True,
                'result': decision_result
            }
            
            logger.info("Decision: action=%s, confidence=%.2f, reasoning='%s'",
                       decision_result['action'],
                       decision_result['confidence'],
                       '; '.join(decision_result['reasoning']))
            
            # Alert if significant action recommended
            if decision_result['action'] in ['retrain', 'rollback', 'investigate']:
                self.alert_system.send_alert(
                    title=f"Action Recommended: {decision_result['action']}",
                    message='; '.join(decision_result['reasoning']),
                    severity="WARNING" if decision_result['action'] == 'investigate' else "CRITICAL",
                    context={
                        'action': decision_result['action'],
                        'confidence': decision_result['confidence'],
                        'health_severity': decision_result['health_severity'],
                        'drift_severity': decision_result['drift_severity']
                    }
                )
            
            return decision_result
            
        except Exception as e:
            logger.error("Decision making stage failed: %s", str(e))
            report['stages']['decision_making'] = {'success': False, 'error': str(e)}
            raise
    
    def _stage_execute_action(
        self,
        decision: Dict[str, Any],
        current_model: Any,
        X_train: Optional[pd.DataFrame],
        y_train: Optional[pd.Series],
        X_val: pd.DataFrame,
        y_val: pd.Series,
        dry_run: bool,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the decided action."""
        logger.info("Stage 4: Executing Action - %s", decision['action'])
        
        action = decision['action']
        action_result = {'action': action, 'new_model': None}
        
        try:
            if action == "retrain":
                if dry_run:
                    logger.info("[DRY RUN] Would retrain model")
                    action_result['dry_run'] = True
                elif X_train is None or y_train is None:
                    logger.warning("Retraining requested but no training data provided")
                    action_result['skipped'] = True
                    action_result['reason'] = 'no_training_data'
                else:
                    # Retrain model
                    new_model = self.retrainer.retrain(
                        X_train, y_train,
                        validation_data=(X_val, y_val),
                        base_model=current_model
                    )
                    
                    # Save new version
                    version_id = self.rollback_manager.save_version(
                        new_model,
                        version_tag='retrained',
                        metadata={'trigger': 'self_healing', 'decision': decision}
                    )
                    
                    action_result['new_model'] = new_model
                    action_result['version_id'] = version_id
                    logger.info("Model retrained and saved as version: %s", version_id)
            
            elif action == "rollback":
                if dry_run:
                    logger.info("[DRY RUN] Would rollback to previous version")
                    action_result['dry_run'] = True
                else:
                    # Rollback to stable version
                    previous_model = self.rollback_manager.rollback(
                        version_tag='stable',
                        validation_data=(X_val, y_val)
                    )
                    
                    action_result['new_model'] = previous_model
                    logger.info("Rolled back to stable version")
            
            elif action in ["investigate", "alert"]:
                # These actions don't change the model, just notify
                logger.info("Action '%s' requires manual intervention", action)
                action_result['requires_manual_intervention'] = True
            
            else:  # no_action, monitor
                logger.info("No model changes required")
                action_result['no_changes'] = True
            
            report['stages']['action_execution'] = {
                'success': True,
                'result': action_result
            }
            
            return action_result
            
        except Exception as e:
            logger.error("Action execution failed: %s", str(e))
            report['stages']['action_execution'] = {'success': False, 'error': str(e)}
            raise
    
    def _stage_validation(
        self,
        new_model: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate new model."""
        logger.info("Stage 5: Validation")
        
        try:
            y_pred = new_model.predict(X_val)
            y_pred_proba = new_model.predict_proba(X_val) if hasattr(new_model, 'predict_proba') else None
            
            validation_result = self.health_monitor.evaluate(y_val, y_pred, y_pred_proba)
            
            report['stages']['validation'] = {
                'success': True,
                'result': validation_result
            }
            
            logger.info("Validation: is_healthy=%s, recall=%.4f",
                       validation_result['is_healthy'],
                       validation_result['metrics']['recall'])
            
            return validation_result
            
        except Exception as e:
            logger.error("Validation stage failed: %s", str(e))
            report['stages']['validation'] = {'success': False, 'error': str(e)}
            raise
    
    def _transition_state(self, new_state: HealingState) -> None:
        """Transition to a new workflow state."""
        logger.debug("State transition: %s -> %s", self.current_state.value, new_state.value)
        self.current_state = new_state
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of all workflows."""
        if not self.workflow_history:
            return {'workflows_executed': 0}
        
        successful = sum(1 for w in self.workflow_history if w['success'])
        
        # Action distribution
        actions = {}
        for workflow in self.workflow_history:
            action = workflow.get('final_action', 'unknown')
            actions[action] = actions.get(action, 0) + 1
        
        return {
            'workflows_executed': len(self.workflow_history),
            'successful': successful,
            'failed': len(self.workflow_history) - successful,
            'success_rate': successful / len(self.workflow_history),
            'actions_taken': actions,
            'recent_workflows': self.workflow_history[-5:],
            'avg_duration': np.mean([w['duration_seconds'] for w in self.workflow_history if 'duration_seconds' in w]),
        }
