"""
Attack Validation Utilities

Provides input validation and sanity checks for attack simulation framework.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Union, List
from functools import wraps

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_dataframe(
    df: pd.DataFrame,
    name: str = "DataFrame",
    min_rows: int = 1,
    min_cols: int = 1,
    required_cols: Optional[List[str]] = None,
    numeric_only: bool = True
) -> None:
    """
    Validate a pandas DataFrame meets requirements.
    
    Args:
        df: DataFrame to validate
        name: Name for error messages
        min_rows: Minimum number of rows required
        min_cols: Minimum number of columns required
        required_cols: List of required column names
        numeric_only: If True, check all columns are numeric
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(df, pd.DataFrame):
        raise ValidationError(f"{name} must be a pandas DataFrame, got {type(df)}")
    
    if len(df) < min_rows:
        raise ValidationError(f"{name} must have at least {min_rows} rows, got {len(df)}")
    
    if len(df.columns) < min_cols:
        raise ValidationError(f"{name} must have at least {min_cols} columns, got {len(df.columns)}")
    
    if required_cols:
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValidationError(f"{name} missing required columns: {missing}")
    
    if numeric_only:
        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric:
            raise ValidationError(f"{name} contains non-numeric columns: {non_numeric}")


def validate_series(
    s: pd.Series,
    name: str = "Series",
    min_length: int = 1,
    allow_negative: bool = True
) -> None:
    """
    Validate a pandas Series.
    
    Args:
        s: Series to validate
        name: Name for error messages
        min_length: Minimum length required
        allow_negative: If False, check all values are non-negative
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(s, pd.Series):
        raise ValidationError(f"{name} must be a pandas Series, got {type(s)}")
    
    if len(s) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} elements, got {len(s)}")
    
    if not allow_negative and (s < 0).any():
        raise ValidationError(f"{name} contains negative values")


def validate_fraction(value: float, name: str = "fraction") -> None:
    """
    Validate a value is between 0 and 1.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if not 0 <= value <= 1:
        raise ValidationError(f"{name} must be between 0 and 1, got {value}")


def validate_positive(value: Union[int, float], name: str = "value") -> None:
    """
    Validate a value is positive.
    
    Args:
        value: Value to validate
        name: Name for error messages
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(value)}")
    
    if value <= 0:
        raise ValidationError(f"{name} must be positive, got {value}")


def validate_config(config: dict, schema: dict) -> None:
    """
    Validate configuration dictionary against a schema.
    
    Args:
        config: Configuration to validate
        schema: Schema dict with {key: (type, validator_func)}
        
    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Config must be a dict, got {type(config)}")
    
    for key, (expected_type, validator) in schema.items():
        if key in config:
            value = config[key]
            if not isinstance(value, expected_type):
                raise ValidationError(
                    f"Config['{key}'] must be {expected_type.__name__}, got {type(value).__name__}"
                )
            if validator:
                validator(value, f"Config['{key}']")


def validate_attack_input(func):
    """
    Decorator to validate attack function inputs.
    
    Validates that X is a DataFrame and performs basic checks.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get X from args or kwargs
        X = None
        if len(args) > 0:
            X = args[0]
        elif 'X' in kwargs:
            X = kwargs['X']
        
        if X is not None:
            try:
                validate_dataframe(X, name=f"{func.__name__}.X", numeric_only=True)
            except ValidationError as e:
                logger.error(f"Validation failed in {func.__name__}: {e}")
                raise
        
        return func(*args, **kwargs)
    
    return wrapper


if __name__ == "__main__":
    # Test validators
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    validate_dataframe(df)
    
    s = pd.Series([1, 2, 3])
    validate_series(s)
    
    validate_fraction(0.5)
    validate_positive(10)
    
    print("All validators passed!")
