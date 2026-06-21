"""
MLflow Integration Module

Provides experiment tracking and model registry capabilities:
- Backtest run logging
- Metric tracking
- Parameter versioning
- Model artifact storage
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import MLflow
try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not installed. Install with: pip install mlflow")


class MLflowTracker:
    """
    MLflow integration for trading strategy tracking.
    
    Tracks:
    - Backtest parameters and metrics
    - Strategy configurations
    - Model versions
    - Optimization runs
    """
    
    def __init__(
        self,
        experiment_name: str = "TradingIA",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None
    ) -> None:
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local file)
            artifact_location: Where to store artifacts
        """
        self.experiment_name = experiment_name
        self.logger = logging.getLogger(__name__)
        self._enabled = MLFLOW_AVAILABLE
        
        if not self._enabled:
            self.logger.warning("MLflow not available, tracking disabled")
            return
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local mlruns directory
            local_path = Path("mlruns")
            local_path.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{local_path.absolute()}")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                self.experiment_id = mlflow.create_experiment(
                    experiment_name,
                    artifact_location=artifact_location
                )
            else:
                self.experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_name)
            self.logger.info(f"MLflow experiment: {experiment_name}")
        except (mlflow.exceptions.MlflowException, OSError) as e:
            self.logger.error(f"Failed to setup MLflow experiment: {e}")
            self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if MLflow tracking is enabled."""
        return self._enabled
    
    def log_backtest(
        self,
        strategy_name: str,
        parameters: Dict,
        metrics: Dict,
        data_info: Optional[Dict] = None,
        tags: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Log a backtest run to MLflow.
        
        Args:
            strategy_name: Name of the strategy
            parameters: Strategy parameters
            metrics: Performance metrics
            data_info: Information about data used
            tags: Additional tags
            
        Returns:
            Run ID if successful, None otherwise
        """
        if not self._enabled:
            return None
        
        try:
            with mlflow.start_run(run_name=f"{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                for key, value in parameters.items():
                    try:
                        mlflow.log_param(key, value)
                    except (TypeError, mlflow.exceptions.MlflowException):
                        # Handle complex objects that can't be serialized directly
                        mlflow.log_param(key, str(value))
                
                # Log metrics
                for key, value in metrics.items():
                    if isinstance(value, (int, float)) and key != 'error':
                        mlflow.log_metric(key, value)
                
                # Log data info
                if data_info:
                    for key, value in data_info.items():
                        mlflow.log_param(f"data_{key}", value)
                
                # Set tags
                mlflow.set_tag("strategy", strategy_name)
                mlflow.set_tag("timestamp", datetime.now().isoformat())
                
                if tags:
                    for key, value in tags.items():
                        mlflow.set_tag(key, value)
                
                run_id = mlflow.active_run().info.run_id
                self.logger.info(f"Logged backtest run: {run_id}")
                return run_id
                
        except (mlflow.exceptions.MlflowException, OSError) as e:
            self.logger.error(f"Failed to log backtest: {e}")
            return None
    
    def log_wfa_run(
        self,
        strategy_name: str,
        wfa_result: Dict,
        param_ranges: Dict
    ) -> Optional[str]:
        """
        Log Walk-Forward Analysis run.
        
        Args:
            strategy_name: Name of the strategy
            wfa_result: WFA results dictionary
            param_ranges: Parameter ranges used for optimization
            
        Returns:
            Run ID if successful
        """
        if not self._enabled:
            return None
        
        try:
            with mlflow.start_run(run_name=f"WFA_{strategy_name}"):
                # Log WFA parameters
                mlflow.log_param("n_periods", len(wfa_result.get('period_results', [])))
                mlflow.log_param("optimization_used", wfa_result.get('optimization_used', False))
                
                # Log param ranges as JSON artifact
                mlflow.log_dict(param_ranges, "param_ranges.json")
                
                # Log WFA metrics
                mlflow.log_metric("avg_degradation", wfa_result.get('avg_degradation', 0))
                mlflow.log_metric("avg_oos_sharpe", wfa_result.get('avg_oos_sharpe', 0))
                mlflow.log_metric("stability_score", wfa_result.get('stability_score', 0))
                mlflow.log_metric("certified", 1 if wfa_result.get('certified', False) else 0)
                
                # Log best params
                best_params = wfa_result.get('best_params', {})
                for key, value in best_params.items():
                    mlflow.log_param(f"best_{key}", value)
                
                # Log period results as artifact
                if wfa_result.get('period_results'):
                    mlflow.log_dict(
                        {'periods': wfa_result['period_results']},
                        "period_results.json"
                    )
                
                # Tags
                mlflow.set_tag("run_type", "wfa")
                mlflow.set_tag("strategy", strategy_name)
                mlflow.set_tag("certified", str(wfa_result.get('certified', False)))
                
                return mlflow.active_run().info.run_id
                
        except Exception as e:
            self.logger.error(f"Failed to log WFA run: {e}")
            return None
    
    def log_monte_carlo(
        self,
        strategy_name: str,
        mc_result: Dict
    ) -> Optional[str]:
        """
        Log Monte Carlo simulation results.
        
        Args:
            strategy_name: Name of the strategy
            mc_result: Monte Carlo results
            
        Returns:
            Run ID if successful
        """
        if not self._enabled:
            return None
        
        try:
            with mlflow.start_run(run_name=f"MC_{strategy_name}"):
                # Log MC metrics
                mlflow.log_metric("sharpe_mean", mc_result.get('sharpe_mean', 0))
                mlflow.log_metric("sharpe_std", mc_result.get('sharpe_std', 0))
                mlflow.log_metric("sharpe_p5", mc_result.get('sharpe_p5', 0))
                mlflow.log_metric("sharpe_p95", mc_result.get('sharpe_p95', 0))
                mlflow.log_metric("win_rate_mean", mc_result.get('win_rate_mean', 0))
                mlflow.log_metric("is_robust", 1 if mc_result.get('is_robust', False) else 0)
                
                # Log simulation params
                mlflow.log_param("num_simulations", mc_result.get('num_simulations', 0))
                
                # Tags
                mlflow.set_tag("run_type", "monte_carlo")
                mlflow.set_tag("strategy", strategy_name)
                mlflow.set_tag("is_robust", str(mc_result.get('is_robust', False)))
                
                return mlflow.active_run().info.run_id
                
        except Exception as e:
            self.logger.error(f"Failed to log Monte Carlo: {e}")
            return None
    
    def log_model_version(
        self,
        strategy_name: str,
        version_info: Dict,
        model_path: Optional[str] = None
    ) -> Optional[str]:
        """
        Log a model version.
        
        Args:
            strategy_name: Name of the strategy
            version_info: Version information (from RetrainingPipeline)
            model_path: Path to model artifacts
            
        Returns:
            Run ID if successful
        """
        if not self._enabled:
            return None
        
        try:
            with mlflow.start_run(run_name=f"Model_{strategy_name}"):
                # Log version info
                mlflow.log_param("version_id", version_info.get('version_id', ''))
                mlflow.log_param("trained_on_bars", version_info.get('trained_on_bars', 0))
                mlflow.log_param("trigger", version_info.get('trigger', 'manual'))
                
                # Log parameters
                params = version_info.get('parameters', {})
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Log metrics
                metrics = version_info.get('metrics', {})
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        mlflow.log_metric(key, value)
                
                mlflow.log_metric("validation_sharpe", version_info.get('validation_sharpe', 0))
                
                # Log model artifacts
                if model_path and os.path.exists(model_path):
                    mlflow.log_artifacts(model_path)
                
                # Tags
                mlflow.set_tag("run_type", "model_version")
                mlflow.set_tag("strategy", strategy_name)
                mlflow.set_tag("version_id", version_info.get('version_id', ''))
                
                return mlflow.active_run().info.run_id
                
        except Exception as e:
            self.logger.error(f"Failed to log model version: {e}")
            return None
    
    def get_best_run(
        self,
        strategy_name: str,
        metric: str = "sharpe",
        ascending: bool = False
    ) -> Optional[Dict]:
        """
        Get the best run for a strategy based on a metric.
        
        Args:
            strategy_name: Name of the strategy
            metric: Metric to optimize
            ascending: If True, lower is better
            
        Returns:
            Best run info or None
        """
        if not self._enabled:
            return None
        
        try:
            client = MlflowClient()
            
            order = "ASC" if ascending else "DESC"
            runs = client.search_runs(
                experiment_ids=[self.experiment_id],
                filter_string=f"tags.strategy = '{strategy_name}'",
                order_by=[f"metrics.{metric} {order}"],
                max_results=1
            )
            
            if runs:
                run = runs[0]
                return {
                    'run_id': run.info.run_id,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'tags': run.data.tags
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get best run: {e}")
            return None
    
    def compare_runs(
        self,
        run_ids: List[str]
    ) -> Optional[List[Dict]]:
        """
        Compare multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            
        Returns:
            List of run comparisons
        """
        if not self._enabled:
            return None
        
        try:
            client = MlflowClient()
            comparisons = []
            
            for run_id in run_ids:
                run = client.get_run(run_id)
                comparisons.append({
                    'run_id': run_id,
                    'metrics': run.data.metrics,
                    'params': run.data.params,
                    'start_time': run.info.start_time
                })
            
            return comparisons
            
        except Exception as e:
            self.logger.error(f"Failed to compare runs: {e}")
            return None


# Singleton instance
_tracker: Optional[MLflowTracker] = None


def get_tracker(
    experiment_name: str = "TradingIA",
    tracking_uri: Optional[str] = None
) -> MLflowTracker:
    """
    Get or create global MLflow tracker instance.
    
    Args:
        experiment_name: Name of the experiment
        tracking_uri: MLflow tracking URI
        
    Returns:
        MLflowTracker instance
    """
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker(experiment_name, tracking_uri)
    return _tracker
