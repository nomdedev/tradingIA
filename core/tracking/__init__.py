"""
Tracking module for experiment and model tracking.

Provides:
- MLflow integration for experiment tracking
- Metric logging and comparison
- Model versioning
"""

from core.tracking.mlflow_tracker import (
    MLflowTracker,
    get_tracker,
    MLFLOW_AVAILABLE
)

__all__ = [
    'MLflowTracker',
    'get_tracker',
    'MLFLOW_AVAILABLE'
]
