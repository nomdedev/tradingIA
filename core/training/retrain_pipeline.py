"""
Automatic Retraining Pipeline Module

Provides automated strategy retraining capabilities:
- Performance degradation detection
- Scheduled retraining triggers
- Parameter versioning and rollback
- A/B testing for new models
"""

import os
import json
import logging
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class RetrainTrigger(Enum):
    """Reasons for triggering retraining."""
    SCHEDULED = "scheduled"
    DEGRADATION = "degradation"
    MANUAL = "manual"
    DRIFT = "drift"
    NEW_DATA = "new_data"


@dataclass
class ModelVersion:
    """Represents a specific model/parameter version."""
    version_id: str
    strategy_name: str
    parameters: Dict
    metrics: Dict
    created_at: str
    trained_on_bars: int
    is_active: bool
    trigger: RetrainTrigger
    validation_sharpe: float
    notes: str = ""
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['trigger'] = self.trigger.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelVersion':
        data['trigger'] = RetrainTrigger(data['trigger'])
        return cls(**data)


@dataclass
class RetrainConfig:
    """Configuration for automatic retraining."""
    # Degradation thresholds
    sharpe_degradation_threshold: float = 0.3  # 30% drop triggers retrain
    win_rate_degradation_threshold: float = 0.1  # 10% drop
    consecutive_losses_trigger: int = 10
    
    # Scheduling
    retrain_interval_days: int = 30
    min_bars_between_retrains: int = 1000
    
    # Validation
    validation_split: float = 0.2
    min_validation_sharpe: float = 0.5
    
    # Rollback
    max_versions_to_keep: int = 10
    auto_rollback_on_degradation: bool = True
    
    # A/B Testing
    ab_test_allocation: float = 0.1  # 10% of capital for testing


class RetrainingPipeline:
    """
    Automated Strategy Retraining Pipeline.
    
    Features:
    - Continuous performance monitoring
    - Automatic trigger detection
    - Parameter optimization and validation
    - Version control with rollback capability
    - A/B testing for gradual rollouts
    """
    
    def __init__(
        self,
        models_dir: str = "models",
        config: Optional[RetrainConfig] = None,
        optimize_func: Optional[Callable] = None,
        backtest_func: Optional[Callable] = None
    ) -> None:
        """
        Initialize retraining pipeline.
        
        Args:
            models_dir: Directory for storing model versions
            config: Retraining configuration
            optimize_func: Function for parameter optimization
            backtest_func: Function for backtesting strategies
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = config or RetrainConfig()
        self.optimize_func = optimize_func
        self.backtest_func = backtest_func
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._performance_history: Dict[str, List[Dict]] = {}
        self._active_versions: Dict[str, ModelVersion] = {}
        
        # Load existing versions
        self._load_versions()
    
    def _load_versions(self) -> None:
        """Load existing model versions from disk."""
        versions_file = self.models_dir / "versions.json"
        
        if versions_file.exists():
            try:
                with open(versions_file, 'r') as f:
                    data = json.load(f)
                
                for strategy, version_data in data.get('active_versions', {}).items():
                    self._active_versions[strategy] = ModelVersion.from_dict(version_data)
                
                self.logger.info(f"Loaded {len(self._active_versions)} active versions")
            except Exception as e:
                self.logger.error(f"Error loading versions: {e}")
    
    def _save_versions(self) -> None:
        """Save model versions to disk."""
        versions_file = self.models_dir / "versions.json"
        
        data = {
            'active_versions': {
                name: version.to_dict() 
                for name, version in self._active_versions.items()
            },
            'last_updated': datetime.now().isoformat()
        }
        
        with open(versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def check_retrain_needed(
        self,
        strategy_name: str,
        current_metrics: Dict,
        recent_trades: Optional[pd.DataFrame] = None
    ) -> Tuple[bool, Optional[RetrainTrigger], str]:
        """
        Check if retraining is needed for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            current_metrics: Current performance metrics
            recent_trades: Recent trade history
            
        Returns:
            Tuple of (needs_retrain, trigger_type, reason)
        """
        active_version = self._active_versions.get(strategy_name)
        
        if active_version is None:
            return True, RetrainTrigger.NEW_DATA, "No existing version found"
        
        # Check scheduled retrain
        created_at = datetime.fromisoformat(active_version.created_at)
        days_since_train = (datetime.now() - created_at).days
        
        if days_since_train >= self.config.retrain_interval_days:
            return True, RetrainTrigger.SCHEDULED, f"Scheduled retrain ({days_since_train} days)"
        
        # Check performance degradation
        baseline_sharpe = active_version.validation_sharpe
        current_sharpe = current_metrics.get('sharpe', 0)
        
        if baseline_sharpe > 0:
            sharpe_degradation = (baseline_sharpe - current_sharpe) / baseline_sharpe
            if sharpe_degradation > self.config.sharpe_degradation_threshold:
                return True, RetrainTrigger.DEGRADATION, f"Sharpe degraded {sharpe_degradation*100:.1f}%"
        
        # Check win rate degradation
        baseline_wr = active_version.metrics.get('win_rate', 0.5)
        current_wr = current_metrics.get('win_rate', 0)
        
        if baseline_wr > 0:
            wr_degradation = (baseline_wr - current_wr) / baseline_wr
            if wr_degradation > self.config.win_rate_degradation_threshold:
                return True, RetrainTrigger.DEGRADATION, f"Win rate degraded {wr_degradation*100:.1f}%"
        
        # Check consecutive losses
        if recent_trades is not None and 'pnl' in recent_trades.columns:
            recent_results = recent_trades['pnl'].tail(self.config.consecutive_losses_trigger)
            if (recent_results < 0).all():
                return True, RetrainTrigger.DEGRADATION, f"{len(recent_results)} consecutive losses"
        
        return False, None, "No retrain needed"
    
    def run_retrain(
        self,
        strategy_name: str,
        strategy_class: Any,
        df_multi_tf: Dict[str, pd.DataFrame],
        param_ranges: Dict,
        trigger: RetrainTrigger = RetrainTrigger.MANUAL,
        notes: str = ""
    ) -> Optional[ModelVersion]:
        """
        Run retraining for a strategy.
        
        Args:
            strategy_name: Name of the strategy
            strategy_class: Strategy class to train
            df_multi_tf: Multi-timeframe data
            param_ranges: Parameter ranges for optimization
            trigger: Reason for retraining
            notes: Optional notes
            
        Returns:
            New ModelVersion if successful, None otherwise
        """
        if self.optimize_func is None or self.backtest_func is None:
            self.logger.error("optimize_func and backtest_func required for retraining")
            return None
        
        self.logger.info(f"🔄 Starting retrain for {strategy_name} (trigger: {trigger.value})")
        
        try:
            # Split data for validation
            primary_tf = '5min' if '5min' in df_multi_tf else list(df_multi_tf.keys())[0]
            total_bars = len(df_multi_tf[primary_tf])
            split_idx = int(total_bars * (1 - self.config.validation_split))
            
            train_data = {tf: df.iloc[:split_idx] for tf, df in df_multi_tf.items()}
            val_data = {tf: df.iloc[split_idx:] for tf, df in df_multi_tf.items()}
            
            # Optimize parameters on training data
            self.logger.info(f"   🔍 Optimizing on {split_idx} bars...")
            best_params = self.optimize_func(strategy_class, train_data, param_ranges)
            
            # Validate on holdout
            self.logger.info(f"   📊 Validating on {total_bars - split_idx} bars...")
            val_result = self.backtest_func(strategy_class, val_data, best_params)
            
            if "error" in val_result:
                self.logger.error(f"   ❌ Validation failed: {val_result['error']}")
                return None
            
            val_sharpe = val_result['metrics']['sharpe']
            
            # Check minimum validation threshold
            if val_sharpe < self.config.min_validation_sharpe:
                self.logger.warning(
                    f"   ⚠️ Validation Sharpe {val_sharpe:.2f} below threshold "
                    f"{self.config.min_validation_sharpe}"
                )
                
                # Check if worse than current version
                current = self._active_versions.get(strategy_name)
                if current and val_sharpe < current.validation_sharpe:
                    self.logger.warning("   ⚠️ New version worse than current, keeping existing")
                    return None
            
            # Create new version
            version_id = self._generate_version_id(strategy_name, best_params)
            new_version = ModelVersion(
                version_id=version_id,
                strategy_name=strategy_name,
                parameters=best_params,
                metrics=val_result['metrics'],
                created_at=datetime.now().isoformat(),
                trained_on_bars=split_idx,
                is_active=True,
                trigger=trigger,
                validation_sharpe=val_sharpe,
                notes=notes
            )
            
            # Archive old version
            old_version = self._active_versions.get(strategy_name)
            if old_version:
                self._archive_version(old_version)
            
            # Save new version
            self._save_version_files(new_version)
            self._active_versions[strategy_name] = new_version
            self._save_versions()
            
            self.logger.info(
                f"   ✅ New version {version_id} created "
                f"(Sharpe: {val_sharpe:.2f})"
            )
            
            return new_version
            
        except Exception as e:
            self.logger.error(f"   ❌ Retrain failed: {e}")
            return None
    
    def _generate_version_id(self, strategy_name: str, params: Dict) -> str:
        """Generate unique version ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        params_hash = hashlib.md5(
            json.dumps(params, sort_keys=True).encode()
        ).hexdigest()[:8]
        return f"{strategy_name}_{timestamp}_{params_hash}"
    
    def _save_version_files(self, version: ModelVersion) -> None:
        """Save version parameters to separate file."""
        version_dir = self.models_dir / version.strategy_name
        version_dir.mkdir(parents=True, exist_ok=True)
        
        version_file = version_dir / f"{version.version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _archive_version(self, version: ModelVersion) -> None:
        """Archive an old version."""
        version.is_active = False
        
        archive_dir = self.models_dir / version.strategy_name / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        archive_file = archive_dir / f"{version.version_id}.json"
        with open(archive_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
        
        # Cleanup old archives
        self._cleanup_old_archives(version.strategy_name)
    
    def _cleanup_old_archives(self, strategy_name: str) -> None:
        """Keep only last N archived versions."""
        archive_dir = self.models_dir / strategy_name / "archive"
        
        if not archive_dir.exists():
            return
        
        archives = sorted(archive_dir.glob("*.json"), key=os.path.getmtime)
        
        while len(archives) > self.config.max_versions_to_keep:
            oldest = archives.pop(0)
            oldest.unlink()
            self.logger.debug(f"Removed old archive: {oldest.name}")
    
    def rollback(self, strategy_name: str) -> Optional[ModelVersion]:
        """
        Rollback to previous version.
        
        Args:
            strategy_name: Strategy to rollback
            
        Returns:
            Restored version if successful, None otherwise
        """
        archive_dir = self.models_dir / strategy_name / "archive"
        
        if not archive_dir.exists():
            self.logger.error(f"No archives found for {strategy_name}")
            return None
        
        # Find most recent archive
        archives = sorted(archive_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        
        if not archives:
            self.logger.error(f"No archived versions for {strategy_name}")
            return None
        
        try:
            with open(archives[0], 'r') as f:
                version_data = json.load(f)
            
            restored_version = ModelVersion.from_dict(version_data)
            restored_version.is_active = True
            
            # Archive current version
            current = self._active_versions.get(strategy_name)
            if current:
                self._archive_version(current)
            
            # Restore
            self._active_versions[strategy_name] = restored_version
            self._save_version_files(restored_version)
            self._save_versions()
            
            self.logger.info(
                f"✅ Rolled back {strategy_name} to {restored_version.version_id}"
            )
            
            return restored_version
            
        except Exception as e:
            self.logger.error(f"Rollback failed: {e}")
            return None
    
    def get_active_version(self, strategy_name: str) -> Optional[ModelVersion]:
        """Get the currently active version for a strategy."""
        return self._active_versions.get(strategy_name)
    
    def get_version_history(self, strategy_name: str) -> List[ModelVersion]:
        """Get all versions (active + archived) for a strategy."""
        versions = []
        
        # Add active version
        active = self._active_versions.get(strategy_name)
        if active:
            versions.append(active)
        
        # Add archived versions
        archive_dir = self.models_dir / strategy_name / "archive"
        if archive_dir.exists():
            for archive_file in sorted(archive_dir.glob("*.json"), reverse=True):
                try:
                    with open(archive_file, 'r') as f:
                        version_data = json.load(f)
                    versions.append(ModelVersion.from_dict(version_data))
                except Exception:
                    continue
        
        return versions
    
    def log_performance(
        self,
        strategy_name: str,
        metrics: Dict,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Log performance metrics for monitoring.
        
        Args:
            strategy_name: Strategy name
            metrics: Performance metrics
            timestamp: Optional timestamp (default: now)
        """
        if strategy_name not in self._performance_history:
            self._performance_history[strategy_name] = []
        
        entry = {
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'metrics': metrics
        }
        
        self._performance_history[strategy_name].append(entry)
        
        # Keep last 1000 entries
        if len(self._performance_history[strategy_name]) > 1000:
            self._performance_history[strategy_name] = \
                self._performance_history[strategy_name][-1000:]
    
    def get_performance_trend(
        self,
        strategy_name: str,
        metric: str = 'sharpe',
        lookback: int = 100
    ) -> Optional[Dict]:
        """
        Get performance trend analysis.
        
        Args:
            strategy_name: Strategy name
            metric: Metric to analyze
            lookback: Number of recent entries
            
        Returns:
            Dict with trend analysis
        """
        history = self._performance_history.get(strategy_name, [])
        
        if not history:
            return None
        
        recent = history[-lookback:]
        values = [
            entry['metrics'].get(metric, 0) 
            for entry in recent 
            if metric in entry['metrics']
        ]
        
        if not values:
            return None
        
        values_array = np.array(values)
        
        # Calculate trend
        if len(values) >= 2:
            x = np.arange(len(values))
            slope = np.polyfit(x, values_array, 1)[0]
        else:
            slope = 0
        
        return {
            'metric': metric,
            'current': round(values[-1], 4),
            'mean': round(values_array.mean(), 4),
            'std': round(values_array.std(), 4),
            'min': round(values_array.min(), 4),
            'max': round(values_array.max(), 4),
            'trend_slope': round(slope, 6),
            'trend_direction': 'up' if slope > 0.001 else ('down' if slope < -0.001 else 'flat'),
            'samples': len(values)
        }


# Convenience functions
def create_default_pipeline(
    models_dir: str = "models",
    optimize_func: Optional[Callable] = None,
    backtest_func: Optional[Callable] = None
) -> RetrainingPipeline:
    """Create pipeline with default configuration."""
    return RetrainingPipeline(
        models_dir=models_dir,
        config=RetrainConfig(),
        optimize_func=optimize_func,
        backtest_func=backtest_func
    )
