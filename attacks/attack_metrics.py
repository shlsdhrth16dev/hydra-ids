"""
Attack Effectiveness Metrics

Measures the impact and effectiveness of adversarial attacks on ML models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AttackMetrics:
    """
    Calculate and track attack effectiveness metrics.
    """
    
    def __init__(self, model, scaler=None):
        """
        Initialize metrics calculator.
        
        Args:
            model: Trained ML model with predict/predict_proba methods
            scaler: Optional scaler for preprocessing
        """
        self.model = model
        self.scaler = scaler
    
    def evaluate_attack_impact(
        self,
        X_clean: pd.DataFrame,
        X_attacked: pd.DataFrame,
        y_true: pd.Series,
        attack_metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluate the impact of an attack on model performance.
        
        Args:
            X_clean: Original clean features
            X_attacked: Attacked features
            y_true: True labels
            attack_metadata: Attack metadata from attack function
            
        Returns:
            Dictionary with comprehensive attack metrics
        """
        try:
            logger.info("Evaluating attack impact...")
            
            # Preprocess if scaler available
            if self.scaler:
                X_clean_scaled = self.scaler.transform(X_clean)
                X_attacked_scaled = self.scaler.transform(X_attacked)
            else:
                X_clean_scaled = X_clean.values
                X_attacked_scaled = X_attacked.values
            
            # Get predictions
            y_pred_clean = self.model.predict(X_clean_scaled)
            y_pred_attacked = self.model.predict(X_attacked_scaled)
            
            # Get probabilities
            y_proba_clean = self.model.predict_proba(X_clean_scaled)
            y_proba_attacked = self.model.predict_proba(X_attacked_scaled)
            
            # Calculate metrics for clean data
            clean_metrics = self._calculate_metrics(y_true, y_pred_clean, y_proba_clean, "clean")
            
            # Calculate metrics for attacked data
            attacked_metrics = self._calculate_metrics(y_true, y_pred_attacked, y_proba_attacked, "attacked")
            
            # Calculate degradation
            degradation = {
                'accuracy_drop': clean_metrics['accuracy'] - attacked_metrics['accuracy'],
                'f1_drop': clean_metrics['f1_weighted'] - attacked_metrics['f1_weighted'],
                'precision_drop': clean_metrics['precision_weighted'] - attacked_metrics['precision_weighted'],
                'recall_drop': clean_metrics['recall_weighted'] - attacked_metrics['recall_weighted'],
                'roc_auc_drop': (clean_metrics.get('roc_auc') or 0) - (attacked_metrics.get('roc_auc') or 0)
            }
            
            # Prediction flip analysis
            prediction_changes = (y_pred_clean != y_pred_attacked).sum()
            flip_rate = prediction_changes / len(y_true)
            
            # False negative rate increase (attacks becoming undetected)
            # Assuming 0 is benign, any other class is attack
            fn_clean = ((y_true != 0) & (y_pred_clean == 0)).sum()
            fn_attacked = ((y_true != 0) & (y_pred_attacked == 0)).sum()
            fn_increase = fn_attacked - fn_clean
            
            # Evasion success rate (malicious samples misclassified as benign)
            evasion_success = 0
            if (y_true != 0).sum() > 0:
                evasion_success = fn_attacked / (y_true != 0).sum()
            
            results = {
                'attack_metadata': attack_metadata,
                'clean_performance': clean_metrics,
                'attacked_performance': attacked_metrics,
                'model_degradation': degradation,
                'prediction_analysis': {
                    'predictions_changed': int(prediction_changes),
                    'prediction_flip_rate': float(flip_rate),
                    'false_negatives_increase': int(fn_increase),
                    'evasion_success_rate': float(evasion_success)
                },
                'attack_effectiveness': self._calculate_attack_effectiveness(degradation)
            }
            
            logger.info(f"Attack effectiveness: {results['attack_effectiveness']:.3f}")
            logger.info(f"Accuracy drop: {degradation['accuracy_drop']:.4f}")
            logger.info(f"Prediction flip rate: {flip_rate:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Attack impact evaluation failed: {e}", exc_info=True)
            raise
    
    def _calculate_metrics(
        self,
        y_true,
        y_pred,
        y_proba,
        label: str
    ) -> Dict[str, float]:
        """Calculate standard ML metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        metrics['precision_weighted'] = float(precision)
        metrics['recall_weighted'] = float(recall)
        metrics['f1_weighted'] = float(f1)
        
        # ROC-AUC (if applicable)
        try:
            if len(np.unique(y_true)) == 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(
                    y_true, y_proba, multi_class='ovr', average='weighted'
                ))
        except Exception:
            metrics['roc_auc'] = None
        
        logger.debug(f"{label} metrics: {metrics}")
        
        return metrics
    
    def _calculate_attack_effectiveness(self, degradation: Dict[str, float]) -> float:
        """
        Calculate overall attack effectiveness score (0-1).
        
        Higher score means more effective attack.
        """
        # Weighted combination of metrics
        weights = {
            'accuracy_drop': 0.3,
            'f1_drop': 0.3,
            'precision_drop': 0.2,
            'recall_drop': 0.2
        }
        
        effectiveness = 0
        for metric, weight in weights.items():
            # Normalize to 0-1 and clip
            normalized = max(0, min(1, degradation.get(metric, 0)))
            effectiveness += normalized * weight
        
        return effectiveness
    
    def compare_attacks(
        self,
        attack_results: Dict[str, Dict]
    ) -> pd.DataFrame:
        """
        Compare multiple attack results.
        
        Args:
            attack_results: Dict mapping attack names to their evaluation results
            
        Returns:
            DataFrame with comparative metrics
        """
        comparison_data = []
        
        for attack_name, results in attack_results.items():
            row = {
                'attack': attack_name,
                'attack_type': results.get('attack_metadata', {}).get('attack_type', 'unknown'),
                'effectiveness': results.get('attack_effectiveness', 0),
                'accuracy_drop': results['model_degradation']['accuracy_drop'],
                'f1_drop': results['model_degradation']['f1_drop'],
                'prediction_flip_rate': results['prediction_analysis']['prediction_flip_rate'],
                'evasion_success_rate': results['prediction_analysis']['evasion_success_rate']
            }
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('effectiveness', ascending=False)
        
        return df


if __name__ == "__main__":
    print("Attack metrics module loaded successfully")
