"""
CICIDS Dataset Preprocessing Pipeline

This module provides comprehensive data preprocessing for the CICIDS network intrusion dataset:
- Data cleaning and validation
- Feature engineering for network traffic
- Feature selection and dimensionality reduction
- Stratified train/validation/test splitting
- Feature scaling and transformation
- Metadata tracking

Author: ML Engineering Team
Date: 2026-01-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List, Dict, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import joblib
import json
from datetime import datetime
import sys


# Setup logging
LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / 'preprocessing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class CICIDSPreprocessor:
    """
    Comprehensive preprocessing pipeline for CICIDS network intrusion dataset.
    
    Handles data cleaning, feature engineering, splitting, and scaling.
    """
    
    def __init__(
        self,
        input_file: Path = Path("data/processed/cicids_merged.csv"),
        output_dir: Path = Path("data/processed"),
        test_size: float = 0.15,
        val_size: float = 0.15,
        random_state: int = 42
    ):
        """
        Initialize the preprocessor.
        
        Args:
            input_file: Path to merged CSV file
            output_dir: Directory to save processed data
            test_size: Proportion of data for test set
            val_size: Proportion of training data for validation set
            random_state: Random seed for reproducibility
        """
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Components to be fitted
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.label_names = None
        
        # Metadata
        self.preprocessing_stats = {}
        
        logger.info("Initialized CICIDSPreprocessor")
        logger.info(f"Input file: {self.input_file}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Train/Val/Test split: {1-test_size:.2f}/{val_size:.2f}/{test_size:.2f}")
    
    def load_data(self, sample_size: Optional[int] = None) -> pd.DataFrame:
        """
        Load the merged CICIDS dataset.
        
        Args:
            sample_size: Optional number of rows to sample (for testing)
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from {self.input_file}")
        
        if sample_size:
            logger.info(f"Sampling {sample_size:,} rows for testing")
            # Load in chunks and sample
            chunks = []
            for chunk in pd.read_csv(self.input_file, chunksize=100000):
                chunks.append(chunk)
                if sum(len(c) for c in chunks) >= sample_size:
                    break
            df = pd.concat(chunks, ignore_index=True).sample(n=min(sample_size, sum(len(c) for c in chunks)), random_state=self.random_state)
        else:
            df = pd.read_csv(self.input_file)
        
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        
        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        self.preprocessing_stats['original_rows'] = len(df)
        self.preprocessing_stats['original_columns'] = len(df.columns)
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataset: handle missing values, infinities, duplicates.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning...")
        
        # Check for label column (case-insensitive)
        label_col = None
        for col in df.columns:
            if col.lower() == 'label':
                label_col = col
                break
        
        if label_col is None:
            raise ValueError("Label column not found in dataset")
        
        logger.info(f"Label column found: '{label_col}'")
        
        # Store original shape
        original_shape = df.shape
        
        # 1. Replace infinities with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        logger.info(f"Replaced {inf_count} infinite values")
        
        # 2. Check missing values
        missing_counts = df.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0]
        if len(missing_cols) > 0:
            logger.info(f"Found missing values in {len(missing_cols)} columns")
            for col, count in missing_cols.items():
                logger.info(f"  {col}: {count} ({count/len(df)*100:.2f}%)")
            
            # Drop columns with >50% missing
            high_missing = missing_cols[missing_cols > len(df) * 0.5].index.tolist()
            if high_missing:
                logger.warning(f"Dropping {len(high_missing)} columns with >50% missing: {high_missing}")
                df = df.drop(columns=high_missing)
            
            # Fill remaining numeric missing values with median
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col].fillna(median_val, inplace=True)
                    logger.info(f"Filled {col} missing values with median: {median_val}")
        
        # 3. Remove duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows")
            df = df.drop_duplicates()
        
        # 4. Remove constant columns (zero variance)
        constant_cols = []
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            logger.info(f"Removing {len(constant_cols)} constant columns: {constant_cols}")
            df = df.drop(columns=constant_cols)
        
        logger.info(f"Data cleaning complete: {original_shape} -> {df.shape}")
        
        self.preprocessing_stats['cleaned_rows'] = len(df)
        self.preprocessing_stats['cleaned_columns'] = len(df.columns)
        self.preprocessing_stats['duplicates_removed'] = duplicates
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features for network traffic analysis.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features...")
        
        # Identify potential feature columns (case-insensitive matching)
        def find_col(pattern: str) -> Optional[str]:
            for col in df.columns:
                if pattern.lower() in col.lower():
                    return col
            return None
        
        # Feature engineering based on common CICIDS columns
        new_features = 0
        
        # 1. Packet rate features
        fwd_packets_col = find_col('fwd_packets')
        bwd_packets_col = find_col('bwd_packets')
        flow_duration_col = find_col('flow_duration')
        
        if fwd_packets_col and flow_duration_col:
            df['fwd_packet_rate'] = df[fwd_packets_col] / (df[flow_duration_col] + 1)
            new_features += 1
        
        if bwd_packets_col and flow_duration_col:
            df['bwd_packet_rate'] = df[bwd_packets_col] / (df[flow_duration_col] + 1)
            new_features += 1
        
        # 2. Byte rate features
        total_fwd_packets_col = find_col('total_fwd_packets')
        total_bwd_packets_col = find_col('total_bwd_packets')
        
        if total_fwd_packets_col and flow_duration_col:
            df['fwd_byte_rate'] = df[total_fwd_packets_col] / (df[flow_duration_col] + 1)
            new_features += 1
        
        if total_bwd_packets_col and flow_duration_col:
            df['bwd_byte_rate'] = df[total_bwd_packets_col] / (df[flow_duration_col] + 1)
            new_features += 1
        
        # 3. Packet size ratios
        fwd_packet_len_col = find_col('fwd_packet_length_mean')
        bwd_packet_len_col = find_col('bwd_packet_length_mean')
        
        if fwd_packet_len_col and bwd_packet_len_col:
            df['packet_size_ratio'] = df[fwd_packet_len_col] / (df[bwd_packet_len_col] + 1)
            new_features += 1
        
        # 4. Flow byte to packet ratio
        total_length_col = find_col('total_length_of_fwd_packets')
        if total_length_col and fwd_packets_col:
            df['bytes_per_packet'] = df[total_length_col] / (df[fwd_packets_col] + 1)
            new_features += 1
        
        logger.info(f"Created {new_features} engineered features")
        
        self.preprocessing_stats['engineered_features'] = new_features
        
        return df
    
    def select_features(self, df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select relevant features and remove highly correlated ones.
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            
        Returns:
            Tuple of (DataFrame with selected features, list of feature names)
        """
        logger.info("Selecting features...")
        
        # Separate features and labels
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Only numeric features
        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        X = X[numeric_cols]
        
        logger.info(f"Starting with {len(numeric_cols)} numeric features")
        
        # 1. Remove low variance features
        selector = VarianceThreshold(threshold=0.01)
        X_selected = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        X = X[selected_features]
        
        removed_low_var = len(numeric_cols) - len(selected_features)
        if removed_low_var > 0:
            logger.info(f"Removed {removed_low_var} low variance features")
        
        # 2. Remove highly correlated features
        corr_matrix = X.corr().abs()
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features with correlation > 0.95
        to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]
        
        if to_drop:
            logger.info(f"Removing {len(to_drop)} highly correlated features (corr > 0.95)")
            X = X.drop(columns=to_drop)
        
        feature_names = X.columns.tolist()
        logger.info(f"Final feature count: {len(feature_names)}")
        
        self.preprocessing_stats['final_feature_count'] = len(feature_names)
        self.preprocessing_stats['removed_low_variance'] = removed_low_var
        self.preprocessing_stats['removed_correlated'] = len(to_drop)
        
        # Reconstruct DataFrame with selected features and label
        result_df = X.copy()
        result_df[label_col] = y
        
        return result_df, feature_names
    
    def split_data(
        self, 
        df: pd.DataFrame, 
        label_col: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Stratified train/validation/test split.
        
        Args:
            df: Input DataFrame
            label_col: Name of label column
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data...")
        
        X = df.drop(columns=[label_col])
        y = df[label_col]
        
        # Log class distribution
        logger.info("Label distribution:")
        for label, count in y.value_counts().items():
            logger.info(f"  {label}: {count:,} ({count/len(y)*100:.2f}%)")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            stratify=y_temp,
            random_state=self.random_state
        )
        
        logger.info(f"Train set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"Val set:   {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
        logger.info(f"Test set:  {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
        
        self.preprocessing_stats['train_size'] = len(X_train)
        self.preprocessing_stats['val_size'] = len(X_val)
        self.preprocessing_stats['test_size'] = len(X_test)
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Scale features using StandardScaler (fit on train, transform all).
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            
        Returns:
            Tuple of scaled (X_train, X_val, X_test)
        """
        logger.info("Scaling features...")
        
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        
        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )
        
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )
        
        # Verify scaling
        logger.info(f"Train set - Mean: {X_train_scaled.mean().mean():.6f}, Std: {X_train_scaled.std().mean():.6f}")
        logger.info(f"Val set   - Mean: {X_val_scaled.mean().mean():.6f}, Std: {X_val_scaled.std().mean():.6f}")
        logger.info(f"Test set  - Mean: {X_test_scaled.mean().mean():.6f}, Std: {X_test_scaled.std().mean():.6f}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def encode_labels(
        self,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Encode string labels to integers.
        
        Args:
            y_train: Training labels
            y_val: Validation labels
            y_test: Test labels
            
        Returns:
            Tuple of encoded (y_train, y_val, y_test)
        """
        logger.info("Encoding labels...")
        
        # Fit encoder on all unique labels
        all_labels = pd.concat([y_train, y_val, y_test])
        self.label_encoder.fit(all_labels)
        
        self.label_names = self.label_encoder.classes_.tolist()
        logger.info(f"Label classes ({len(self.label_names)}): {self.label_names}")
        
        # Transform
        y_train_encoded = pd.Series(
            self.label_encoder.transform(y_train),
            index=y_train.index,
            name='label'
        )
        
        y_val_encoded = pd.Series(
            self.label_encoder.transform(y_val),
            index=y_val.index,
            name='label'
        )
        
        y_test_encoded = pd.Series(
            self.label_encoder.transform(y_test),
            index=y_test.index,
            name='label'
        )
        
        return y_train_encoded, y_val_encoded, y_test_encoded
    
    def save_processed_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series
    ):
        """Save processed datasets and preprocessing artifacts."""
        logger.info("Saving processed data...")
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save datasets
        X_train.to_csv(self.output_dir / "X_train.csv", index=False)
        X_val.to_csv(self.output_dir / "X_val.csv", index=False)
        X_test.to_csv(self.output_dir / "X_test.csv", index=False)
        
        y_train.to_csv(self.output_dir / "y_train.csv", index=False)
        y_val.to_csv(self.output_dir / "y_val.csv", index=False)
        y_test.to_csv(self.output_dir / "y_test.csv", index=False)
        
        logger.info(f"Saved datasets to {self.output_dir}")
        
        # Save preprocessing artifacts
        artifacts_dir = Path("models/preprocessing")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.scaler, artifacts_dir / "scaler.joblib")
        joblib.dump(self.label_encoder, artifacts_dir / "label_encoder.joblib")
        
        # Save feature names
        with open(artifacts_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_names, f, indent=2)
        
        # Save label names
        with open(artifacts_dir / "label_names.json", 'w') as f:
            json.dump(self.label_names, f, indent=2)
        
        # Save preprocessing metadata (convert numpy types to native Python)
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing_stats': {k: int(v) if isinstance(v, (np.integer, np.int64)) else v 
                                   for k, v in self.preprocessing_stats.items()},
            'feature_count': len(self.feature_names),
            'label_count': len(self.label_names),
            'label_names': self.label_names,
            'random_state': self.random_state,
            'test_size': self.test_size,
            'val_size': self.val_size
        }
        
        with open(artifacts_dir / "preprocessing_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved preprocessing artifacts to {artifacts_dir}")
    
    def run(self, sample_size: Optional[int] = None):
        """
        Run the complete preprocessing pipeline.
        
        Args:
            sample_size: Optional sample size for testing
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING PREPROCESSING PIPELINE")
            logger.info("=" * 80)
            
            # 1. Load data
            df = self.load_data(sample_size=sample_size)
            
            # 2. Clean data
            df = self.clean_data(df)
            
            # 3. Engineer features
            df = self.engineer_features(df)
            
            # 4. Find label column
            label_col = None
            for col in df.columns:
                if col.lower() == 'label':
                    label_col = col
                    break
            
            # 5. Feature selection
            df, self.feature_names = self.select_features(df, label_col)
            
            # 6. Split data
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df, label_col)
            
            # 7. Scale features
            X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
                X_train, X_val, X_test
            )
            
            # 8. Encode labels
            y_train_encoded, y_val_encoded, y_test_encoded = self.encode_labels(
                y_train, y_val, y_test
            )
            
            # 9. Save everything
            self.save_processed_data(
                X_train_scaled, X_val_scaled, X_test_scaled,
                y_train_encoded, y_val_encoded, y_test_encoded
            )
            
            logger.info("=" * 80)
            logger.info("PREPROCESSING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Final dataset shapes:")
            logger.info(f"  Train: {X_train_scaled.shape}")
            logger.info(f"  Val:   {X_val_scaled.shape}")
            logger.info(f"  Test:  {X_test_scaled.shape}")
            logger.info(f"  Features: {len(self.feature_names)}")
            logger.info(f"  Labels: {len(self.label_names)}")
            logger.info("=" * 80)
            
            return True
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {str(e)}", exc_info=True)
            return False


def main():
    """Main execution function."""
    # Create preprocessor
    preprocessor = CICIDSPreprocessor(
        input_file=Path("data/processed/cicids_merged.csv"),
        output_dir=Path("data/processed"),
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Run preprocessing
    # Use sample_size=100000 for quick testing, None for full dataset
    success = preprocessor.run(sample_size=None)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
