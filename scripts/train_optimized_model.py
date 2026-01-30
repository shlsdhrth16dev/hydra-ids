"""
Optimized ML Model Training for CICIDS Intrusion Detection

This script trains optimized machine learning models on the preprocessed CICIDS dataset:
- XGBoost with hyperparameter tuning
- Random Forest baseline
- Comprehensive evaluation metrics
- Model persistence and versioning

Author: ML Engineering Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import json
from datetime import datetime
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve
)
from sklearn.model_selection import GridSearchCV
import joblib

# Setup logging
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'model_training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class OptimizedIDSTrainer:
    """
    Optimized ML training pipeline for intrusion detection.
    """
    
    def __init__(
        self,
        data_dir: Path = Path("data/processed"),
        model_dir: Path = Path("models"),
        random_state: int = 42
    ):
        """
        Initialize the trainer.
        
        Args:
            data_dir: Directory containing processed data
            model_dir: Directory to save trained models
            random_state: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.random_state = random_state
        
        # Model storage
        self.models = {}
        self.results = {}
        
        logger.info("Initialized OptimizedIDSTrainer")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Model directory: {self.model_dir}")
    
    def load_data(self):
        """Load preprocessed train/validation/test data."""
        logger.info("Loading preprocessed data...")
        
        # Load features
        self.X_train = pd.read_csv(self.data_dir / "X_train.csv")
        self.X_val = pd.read_csv(self.data_dir / "X_val.csv")
        self.X_test = pd.read_csv(self.data_dir / "X_test.csv")
        
        # Load labels
        self.y_train = pd.read_csv(self.data_dir / "y_train.csv").squeeze()
        self.y_val = pd.read_csv(self.data_dir / "y_val.csv").squeeze()
        self.y_test = pd.read_csv(self.data_dir / "y_test.csv").squeeze()
        
        logger.info(f"Train set: {self.X_train.shape}")
        logger.info(f"Val set:   {self.X_val.shape}")
        logger.info(f"Test set:  {self.X_test.shape}")
        
        # Calculate class weights for imbalanced data
        unique, counts = np.unique(self.y_train, return_counts=True)
        self.class_distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution in training: {self.class_distribution}")
        
        # Calculate scale_pos_weight for binary classification
        if len(unique) == 2:
            neg_count = self.class_distribution[0]
            pos_count = self.class_distribution[1]
            self.scale_pos_weight = neg_count / pos_count
            logger.info(f"Scale pos weight: {self.scale_pos_weight:.2f}")
        else:
            self.scale_pos_weight = None
    
    def train_random_forest(self):
        """Train Random Forest baseline model."""
        logger.info("=" * 80)
        logger.info("Training Random Forest Baseline")
        logger.info("=" * 80)
        
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            n_jobs=-1,
            random_state=self.random_state,
            class_weight='balanced',
            verbose=1
        )
        
        logger.info("Fitting Random Forest...")
        rf_model.fit(self.X_train, self.y_train)
        
        self.models['random_forest'] = rf_model
        logger.info("Random Forest training complete")
    
    def train_xgboost_basic(self):
        """Train basic XGBoost model."""
        logger.info("=" * 80)
        logger.info("Training XGBoost Model")
        logger.info("=" * 80)
        
        xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        
        # Add scale_pos_weight if binary classification
        if self.scale_pos_weight is not None:
            xgb_params['scale_pos_weight'] = self.scale_pos_weight
        
        xgb_model = XGBClassifier(**xgb_params)
        
        logger.info("Fitting XGBoost...")
        xgb_model.fit(
            self.X_train, 
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=True
        )
        
        self.models['xgboost'] = xgb_model
        logger.info("XGBoost training complete")
    
    def train_xgboost_optimized(self):
        """Train optimized XGBoost with hyperparameter tuning."""
        logger.info("=" * 80)
        logger.info("Training Optimized XGBoost with GridSearch")
        logger.info("=" * 80)
        
        # Define parameter grid for tuning
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9]
        }
        
        base_params = {
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'device': 'cpu'
        }
        
        if self.scale_pos_weight is not None:
            base_params['scale_pos_weight'] = self.scale_pos_weight
        
        xgb_model = XGBClassifier(**base_params)
        
        # Use smaller parameter grid for faster execution
        logger.info("Performing GridSearchCV (this may take a while)...")
        grid_search = GridSearchCV(
            estimator=xgb_model,
            param_grid=param_grid,
            scoring='f1_weighted',
            cv=3,
            verbose=2,
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.models['xgboost_optimized'] = grid_search.best_estimator_
        logger.info("Optimized XGBoost training complete")
    
    def evaluate_model(self, model_name: str, model: Any) -> Dict[str, Any]:
        """
        Evaluate a model on test set.
        
        Args:
            model_name: Name of the model
            model: Trained model
            
        Returns:
            Dictionary with evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        y_proba = model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='weighted'
        )
        
        # Calculate ROC-AUC (handle multiclass)
        try:
            if len(np.unique(self.y_test)) == 2:
                roc_auc = roc_auc_score(self.y_test, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(
                    self.y_test, y_proba, multi_class='ovr', average='weighted'
                )
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        
        results = {
            'model_name': model_name,
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc) if roc_auc is not None else None,
            'confusion_matrix': cm.tolist(),
            'classification_report': report
        }
        
        # Log results
        logger.info(f"\n{model_name} Results:")
        logger.info(f"  Accuracy:  {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall:    {recall:.4f}")
        logger.info(f"  F1 Score:  {f1:.4f}")
        if roc_auc is not None:
            logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
        
        logger.info("\nConfusion Matrix:")
        logger.info(f"\n{cm}")
        
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(self.y_test, y_pred)}")
        
        return results
    
    def evaluate_all_models(self):
        """Evaluate all trained models."""
        logger.info("=" * 80)
        logger.info("Evaluating All Models")
        logger.info("=" * 80)
        
        for model_name, model in self.models.items():
            results = self.evaluate_model(model_name, model)
            self.results[model_name] = results
    
    def save_models_and_results(self):
        """Save trained models and evaluation results."""
        logger.info("Saving models and results...")
        
        # Save each model
        for model_name, model in self.models.items():
            model_path = self.model_dir / model_name
            model_path.mkdir(parents=True, exist_ok=True)
            
            model_file = model_path / f"{model_name}_v1.joblib"
            joblib.dump(model, model_file)
            logger.info(f"Saved {model_name} to {model_file}")
        
        # Save results
        results_dir = self.model_dir / "evaluation_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"model_evaluation_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved evaluation results to {results_file}")
        
        # Create summary report
        self.create_summary_report(results_dir / f"summary_report_{timestamp}.md")
    
    def create_summary_report(self, report_path: Path):
        """Create a markdown summary report."""
        logger.info("Creating summary report...")
        
        with open(report_path, 'w') as f:
            f.write("# CICIDS Intrusion Detection Model Evaluation Report\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Information\n")
            f.write(f"- Training samples: {len(self.X_train):,}\n")
            f.write(f"- Validation samples: {len(self.X_val):,}\n")
            f.write(f"- Test samples: {len(self.X_test):,}\n")
            f.write(f"- Features: {self.X_train.shape[1]}\n")
            f.write(f"- Classes: {len(np.unique(self.y_train))}\n\n")
            
            f.write("## Model Performance Summary\n\n")
            f.write("| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |\n")
            f.write("|-------|----------|-----------|--------|----------|----------|\n")
            
            for model_name, results in self.results.items():
                f.write(f"| {model_name} | ")
                f.write(f"{results['accuracy']:.4f} | ")
                f.write(f"{results['precision']:.4f} | ")
                f.write(f"{results['recall']:.4f} | ")
                f.write(f"{results['f1_score']:.4f} | ")
                if results['roc_auc'] is not None:
                    f.write(f"{results['roc_auc']:.4f} |\n")
                else:
                    f.write("N/A |\n")
            
            f.write("\n## Detailed Results\n\n")
            
            for model_name, results in self.results.items():
                f.write(f"### {model_name}\n\n")
                f.write("**Confusion Matrix:**\n")
                f.write("```\n")
                cm = np.array(results['confusion_matrix'])
                f.write(str(cm))
                f.write("\n```\n\n")
        
        logger.info(f"Summary report saved to {report_path}")
    
    def run(self, skip_grid_search: bool = False):
        """
        Run the complete training pipeline.
        
        Args:
            skip_grid_search: If True, skip the expensive grid search optimization
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING MODEL TRAINING PIPELINE")
            logger.info("=" * 80)
            
            # Load data
            self.load_data()
            
            # Train models
            self.train_random_forest()
            self.train_xgboost_basic()
            
            if not skip_grid_search:
                self.train_xgboost_optimized()
            else:
                logger.info("Skipping grid search optimization (use --grid-search to enable)")
            
            # Evaluate all models
            self.evaluate_all_models()
            
            # Save everything
            self.save_models_and_results()
            
            logger.info("=" * 80)
            logger.info("MODEL TRAINING COMPLETE")
            logger.info("=" * 80)
            
            # Find best model
            best_model = max(
                self.results.items(), 
                key=lambda x: x[1]['f1_score']
            )
            logger.info(f"\nBest Model: {best_model[0]}")
            logger.info(f"Best F1 Score: {best_model[1]['f1_score']:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return False


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train optimized IDS models")
    parser.add_argument(
        '--grid-search', 
        action='store_true', 
        help='Enable expensive grid search for XGBoost optimization'
    )
    args = parser.parse_args()
    
    # Create trainer
    trainer = OptimizedIDSTrainer(
        data_dir=Path("data/processed"),
        model_dir=Path("models"),
        random_state=42
    )
    
    # Run training
    success = trainer.run(skip_grid_search=not args.grid_search)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
