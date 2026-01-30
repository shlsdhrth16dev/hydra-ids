"""
IDS Detection Engine via XGBoost Model.

This module provides the interface to load the trained ML model,
preprocess incoming traffic features, and return attack predictions
in real-time.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class IDSEngine:
    def __init__(self, model_path: Path, scaler_path: Path):
        """
        Initialize the IDS Engine.
        
        Args:
            model_path: Path to the trained XGBoost model (.joblib)
            scaler_path: Path to the fitted StandardScaler (.joblib)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load model and scaler from disk."""
        try:
            logger.info(f"Loading model from {self.model_path}...")
            self.model = joblib.load(self.model_path)
            
            logger.info(f"Loading scaler from {self.scaler_path}...")
            self.scaler = joblib.load(self.scaler_path)
            
            logger.info("IDS Engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to load artifacts: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> dict:
        """
        Predict attack class for a batch of network flows.
        
        Args:
            features (pd.DataFrame): Raw feature vector(s).
            
        Returns:
            dict: {
                'predictions': [class_id, ...],
                'probabilities': [[prob_0, prob_1, ...], ...],
                'is_attack': [bool, ...]
            }
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or Scaler not loaded.")

        # Scaling
        try:
            scaled_features = self.scaler.transform(features)
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            raise

        # Prediction
        preds = self.model.predict(scaled_features)
        probs = self.model.predict_proba(scaled_features)
        
        results = {
            'predictions': preds,
            'probabilities': probs,
            'is_attack': preds != 0  # 0 is 'Benign'
        }
        return results

if __name__ == "__main__":
    # Simple test
    MODEL_DIR = Path(".") / "models"
    try:
        engine = IDSEngine(
            model_path=MODEL_DIR / "xgboost/xgboost_v1.joblib",
            scaler_path=MODEL_DIR / "preprocessing/scaler.joblib"
        )
        print("IDS Engine test passed.")
    except Exception as e:
        print(f"IDS Engine test skipped/failed: {e}")
