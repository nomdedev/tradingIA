"""
Safe access utilities for DataFrames and dictionaries.
Provides safe column access and data preparation for Plotly.
"""

from typing import Mapping, Any, List
import pandas as pd
import numpy as np

def safe_get(d: Mapping | pd.Series, key, default=None):
    """
    Safely get a value from a dictionary or Series.

    Args:
        d: Dictionary or pandas Series
        key: Key to access
        default: Default value if key not found

    Returns:
        Value or default
    """
    try:
        if hasattr(d, 'get'):
            return d.get(key, default)
        elif hasattr(d, '__getitem__'):
            return d[key]
        else:
            return default
    except (KeyError, IndexError, TypeError):
        return default

def prepare_timestamps_for_plotly(series_or_array) -> np.ndarray:
    """
    Prepare timestamps for Plotly charts.

    Converts pandas datetime series to numpy array suitable for Plotly.

    Args:
        series_or_array: Pandas Series or array-like with timestamps

    Returns:
        Numpy array of datetime64 suitable for Plotly
    """
    try:
        if isinstance(series_or_array, pd.Series):
            # Ensure it's datetime
            if not pd.api.types.is_datetime64_any_dtype(series_or_array):
                series_or_array = pd.to_datetime(series_or_array, errors='coerce')

            # Convert to numpy array for Plotly
            return series_or_array.values.astype('datetime64[ns]')

        elif hasattr(series_or_array, '__iter__'):
            # Convert to pandas first, then to numpy
            ts_series = pd.to_datetime(series_or_array, errors='coerce')
            return ts_series.values.astype('datetime64[ns]')

        else:
            # Single value
            ts = pd.to_datetime(series_or_array, errors='coerce')
            return np.array([ts]).astype('datetime64[ns]')

    except Exception:
        # Return empty array on error
        return np.array([], dtype='datetime64[ns]')

def ensure_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """
    Ensure DataFrame has required columns, adding missing ones with NaN.

    Args:
        df: Input DataFrame
        required: List of required column names

    Returns:
        DataFrame with all required columns
    """
    try:
        if df.empty:
            # Create DataFrame with required columns
            return pd.DataFrame(columns=required)

        df = df.copy()

        for col in required:
            if col not in df.columns:
                df[col] = np.nan

        return df

    except Exception:
        # Return original DataFrame on error
        return df

def safe_column_access(df: pd.DataFrame, column: str, default_value=None):
    """
    Safely access a DataFrame column.

    Args:
        df: DataFrame to access
        column: Column name
        default_value: Default value if column doesn't exist

    Returns:
        Column values or default
    """
    try:
        if column in df.columns:
            return df[column]
        else:
            if default_value is not None:
                return pd.Series([default_value] * len(df), index=df.index)
            else:
                return pd.Series([np.nan] * len(df), index=df.index)
    except Exception:
        if default_value is not None:
            return pd.Series([default_value] * len(df), index=df.index) if not df.empty else pd.Series()
        else:
            return pd.Series([np.nan] * len(df), index=df.index) if not df.empty else pd.Series()

def validate_dataframe(df: pd.DataFrame, required_columns: List[str] = None) -> bool:
    """
    Validate DataFrame has required structure.

    Args:
        df: DataFrame to validate
        required_columns: List of columns that must exist

    Returns:
        True if valid, False otherwise
    """
    try:
        if df is None or df.empty:
            return False

        if required_columns:
            for col in required_columns:
                if col not in df.columns:
                    return False

        return True

    except Exception:
        return False

def clean_numeric_column(series: pd.Series) -> pd.Series:
    """
    Clean numeric column by converting to float and handling NaN.

    Args:
        series: Input series

    Returns:
        Cleaned numeric series
    """
    try:
        return pd.to_numeric(series, errors='coerce').fillna(0.0)
    except Exception:
        return pd.Series([0.0] * len(series), index=series.index) if not series.empty else pd.Series()

def format_percentage(value: float, decimals: int = 1) -> str:
    """
    Format value as percentage string.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted percentage string
    """
    try:
        return f"{value:.{decimals}f}%"
    except Exception:
        return "0.0%"

def format_currency(value: float, decimals: int = 2) -> str:
    """
    Format value as currency string.

    Args:
        value: Numeric value
        decimals: Number of decimal places

    Returns:
        Formatted currency string
    """
    try:
        return f"${value:,.{decimals}f}"
    except Exception:
        return "$0.00"