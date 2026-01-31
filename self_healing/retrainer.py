"""
Enhanced Retrainer for Adaptive Model Updates.

Provides intelligent model retraining with incremental learning,
hyperparameter optimization, data validation, and comprehensive reporting.
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import xgboost as xgb
import joblib
from datetime import datetime
from pathlib import Path
import logging

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    logging.warning("Optuna not available, hyperparameter tuning disabled")

from .config import RetrainerConfig
from .exceptions import RetrainingError, ValidationError


logger = logging.getLogger(__name__)


class Retrainer:
    """
    Advanced model retraining with multiple strategies.
    
    Features:
    - Full and incremental retraining
    - Multiple model architectures (RF, XGBoost, SGD)
    - Automated hyperparameter optimization
    - Data validation and quality checks
    - Class imbalance handling
    - Training metrics tracking
    - Model comparison and validation
    
    Args:
        config: RetrainerConfig instance
    """
    
    def __init__(self, config: Optional[RetrainerConfig] = None):
        if config is None:
            config = RetrainerConfig()
        
        self.config = config
        self.training_history: List[Dict[str, Any]] = []
        
        logger.info(
            "Retrainer initialized with model_type=%s, incremental=%s",
            config.model_type, config.enable_incremental_learning
        )
    
    def retrain(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
        base_model: Optional[Any] = None
    ) -> Any:
        """
        Train or retrain a model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            validation_data: Optional (X_val, y_val) tuple
            base_model: Base model for incremental learning or warm start
        
        Returns:
            Trained model
        
        Raises:
            RetrainingError: If retraining fails
        """
        try:
            start_time = datetime.now()
            
            # Validate data
            self._validate_training_data(X_train, y_train)
            
            # Handle class imbalance
            class_weights = None
            if self.config.handle_class_imbalance:
                class_weights = self._calculate_class_weights(y_train)
            
            # Choose training strategy
            if self.config.enable_incremental_learning and base_model is not None:
                model = self._incremental_training(X_train, y_train, base_model)
            elif self.config.enable_hyperparameter_tuning and OPTUNA_AVAILABLE:
                model = self._train_with_tuning(X_train, y_train, validation_data, class_weights)
            else:
                model = self._full_training(X_train, y_train, base_model, class_weights)
            
            # Calculate training metrics
            train_score = model.score(X_train, y_train)
            
            val_score = None
            if validation_data is not None:
                X_val, y_val = validation_data
                val_score = model.score(X_val, y_val)
            
            # Record training
            training_time = (datetime.now() - start_time).total_seconds()
            training_record = {
                'timestamp': datetime.now().isoformat(),
                'model_type': self.config.model_type,
                'training_samples': len(X_train),
                'num_features': X_train.shape[1],
                'train_score': float(train_score),
                'val_score': float(val_score) if val_score is not None else None,
                'training_time_seconds': training_time,
                'incremental': self.config.enable_incremental_learning and base_model is not None,
                'hyperparameter_tuning': self.config.enable_hyperparameter_tuning and OPTUNA_AVAILABLE,
            }
            
            self.training_history.append(training_record)
            
            logger.info(
                "Retraining complete: model=%s, train_score=%.4f, val_score=%s, time=%.2fs",
                self.config.model_type, train_score,
                f"{val_score:.4f}" if val_score else "N/A", training_time
            )
            
            return model
            
        except Exception as e:
            logger.error("Retraining failed: %s", str(e), exc_info=True)
            raise RetrainingError(f"Retraining failed: {e}") from e
    
    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Validate training data quality."""
        # Check minimum samples
        if len(X) < self.config.min_training_samples:
            raise ValidationError(
                f"Insufficient training samples: {len(X)} < {self.config.min_training_samples}"
            )
        
        # Check for missing values
        if X.isnull().any().any():
            raise ValidationError("Training data contains missing values")
        
        # Check class imbalance
        class_counts = y.value_counts()
        if len(class_counts) >= 2:
            max_count = class_counts.max()
            min_count = class_counts.min()
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > self.config.max_class_imbalance_ratio:
                logger.warning(
                    "High class imbalance detected: %.2fx (max allowed: %.2fx)",
                    imbalance_ratio, self.config.max_class_imbalance_ratio
                )
        
        logger.info("Data validation passed: %d samples, %d features", len(X), X.shape[1])
    
    def _calculate_class_weights(self, y: pd.Series) -> Dict[int, float]:
        """Calculate class weights for imbalanced data."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        class_weights = dict(zip(classes, weights))
        logger.info("Class weights calculated: %s", class_weights)
        
        return class_weights
    
    def _full_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        base_model: Optional[Any],
        class_weights: Optional[Dict[int, float]]
    ) -> Any:
        """Full model training from scratch or with warm start."""
        if self.config.model_type == "random_forest":
            model = RandomForestClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                class_weight=class_weights or 'balanced',
                random_state=42,
                n_jobs=-1,
                warm_start=self.config.enable_warm_start
            )
            
            # Warm start from base model
            if self.config.enable_warm_start and base_model is not None:
                try:
                    if isinstance(base_model, RandomForestClassifier):
                        model.estimators_ = base_model.estimators_
                        logger.info("Warm start enabled with %d existing estimators", len(model.estimators_))
                except Exception as e:
                    logger.warning("Warm start failed, training from scratch: %s", str(e))
        
        elif self.config.model_type == "xgboost":
            # Convert class weights to scale_pos_weight
            scale_pos_weight = 1.0
            if class_weights and len(class_weights) == 2:
                scale_pos_weight = class_weights[0] / class_weights[1]
            
            model = xgb.XGBClassifier(
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth if self.config.max_depth else 6,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )
        
        elif self.config.model_type == "sgd":
            model = SGDClassifier(
                loss='log_loss',
                class_weight=class_weights or 'balanced',
                random_state=42,
                max_iter=1000,
                tol=1e-3,
            )
        
        else:
            raise RetrainingError(f"Unknown model type: {self.config.model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def _incremental_training(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        base_model: Any
    ) -> Any:
        """Incremental training (partial_fit) for online learning."""
        # Only SGDClassifier supports partial_fit
        if not hasattr(base_model, 'partial_fit'):
            logger.warning(
                "Model type %s doesn't support partial_fit, falling back to full training",
                type(base_model).__name__
            )
            return self._full_training(X_train, y_train, base_model, None)
        
        # Get classes from base model
        classes = getattr(base_model, 'classes_', np.unique(y_train))
        
        # Perform partial fit
        base_model.partial_fit(X_train, y_train, classes=classes)
        
        logger.info("Incremental training completed with %d new samples", len(X_train))
        return base_model
    
    def _train_with_tuning(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        validation_data: Optional[Tuple[pd.DataFrame, pd.Series]],
        class_weights: Optional[Dict[int, float]]
    ) -> Any:
        """Train model with hyperparameter optimization using Optuna."""
        logger.info("Starting hyperparameter optimization with Optuna")
        
        def objective(trial):
            # Suggest hyperparameters based on model type
            if self.config.model_type == "random_forest":
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 5, 30),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                    'class_weight': class_weights or 'balanced',
                    'random_state': 42,
                    'n_jobs': -1,
                }
                model = RandomForestClassifier(**params)
            
            elif self.config.model_type == "xgboost":
                scale_pos_weight = 1.0
                if class_weights and len(class_weights) == 2:
                    scale_pos_weight = class_weights[0] / class_weights[1]
                
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                    'max_depth': trial.suggest_int('max_depth', 3, 15),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'scale_pos_weight': scale_pos_weight,
                    'random_state': 42,
                    'n_jobs': -1,
                }
                model = xgb.XGBClassifier(**params)
            
            else:
                raise RetrainingError(f"Hyperparameter tuning not supported for {self.config.model_type}")
            
            # Train and evaluate
            model.fit(X_train, y_train)
            
            if validation_data is not None:
                X_val, y_val = validation_data
                score = model.score(X_val, y_val)
            else:
                score = model.score(X_train, y_train)
            
            return score
        
        # Run optimization
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=42)
        )
        
        study.optimize(
            objective,
            n_trials=self.config.tuning_trials,
            timeout=self.config.tuning_timeout_seconds,
            show_progress_bar=False
        )
        
        logger.info(
            "Hyperparameter optimization complete: best_score=%.4f, trials=%d",
            study.best_value, len(study.trials)
        )
        
        # Train final model with best parameters
        best_params = study.best_params
        if self.config.model_type == "random_forest":
            best_params['class_weight'] = class_weights or 'balanced'
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            model = RandomForestClassifier(**best_params)
        else:  # xgboost
            scale_pos_weight = 1.0
            if class_weights and len(class_weights) == 2:
                scale_pos_weight = class_weights[0] / class_weights[1]
            best_params['scale_pos_weight'] = scale_pos_weight
            best_params['random_state'] = 42
            best_params['n_jobs'] = -1
            model = xgb.XGBClassifier(**best_params)
        
        model.fit(X_train, y_train)
        return model
    
    def save(self, model: Any, path: str) -> Dict[str, Any]:
        """
        Save model with metadata.
        
        Args:
            model: Model to save
            path: Path to save model
        
        Returns:
            Dictionary with saved model metadata
        """
        try:
            # Create directory if needed
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            joblib.dump(model, path)
            
            # Create metadata
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'model_type': type(model).__name__,
                'path': path,
                'training_history': self.training_history[-1] if self.training_history else None,
            }
            
            # Save metadata alongside model
            metadata_path = path.replace('.joblib', '_metadata.joblib')
            joblib.dump(metadata, metadata_path)
            
            logger.info("Model saved to %s with metadata", path)
            return metadata
            
        except Exception as e:
            logger.error("Failed to save model: %s", str(e), exc_info=True)
            raise RetrainingError(f"Failed to save model: {e}") from e
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training history."""
        if not self.training_history:
            return {'trainings_performed': 0}
        
        return {
            'trainings_performed': len(self.training_history),
            'latest_training': self.training_history[-1],
            'avg_train_score': np.mean([t['train_score'] for t in self.training_history]),
            'avg_training_time': np.mean([t['training_time_seconds'] for t in self.training_history]),
            'incremental_trainings': sum(1 for t in self.training_history if t.get('incremental', False)),
        }
