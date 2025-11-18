"""
Dashboard Controller - Main business logic for the trading dashboard
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from PySide6.QtCore import QObject, Signal, QThread, QTimer
from PySide6.QtWidgets import QApplication

from core.backend_core import DataManager
from core.optimization import OptimizationController
from core.strategies import StrategyRegistry, StrategyConfig
from core.execution.optimized_backtester import OptimizedBacktester
from core.alerts import AlertManager, AlertType, AlertSeverity

logger = logging.getLogger(__name__)

class BacktestWorker(QThread):
    """Worker thread for running backtests asynchronously"""

    finished = Signal(list)  # List of BacktestResult
    progress = Signal(str)  # Progress message
    error = Signal(str)     # Error message

    def __init__(self, data: pd.DataFrame, strategies: List[Any], parameter_sets: List[Dict]):
        super().__init__()
        self.data = data
        self.strategies = strategies
        self.parameter_sets = parameter_sets
        self.backtester = OptimizedBacktester(max_workers=4)

    def run(self):
        """Execute backtests in background thread"""
        try:
            self.progress.emit("Initializing backtests...")

            # Convert strategies to functions for backtester
            strategy_funcs = []
            for strategy in self.strategies:
                strategy_funcs.append(lambda data, **params: strategy.generate_signals(
                    strategy.calculate_indicators(data)
                ))

            self.progress.emit(f"Running {len(strategy_funcs)} parallel backtests...")

            results = self.backtester.run_parallel_backtests(
                self.data, strategy_funcs[0], self.parameter_sets
            )

            self.progress.emit("Backtests completed successfully")
            self.finished.emit(results)

        except Exception as e:
            logger.error(f"Backtest error: {e}")
            self.error.emit(str(e))

class DashboardController(QObject):
    """
    Main controller for the trading dashboard.

    Handles business logic, data management, and communication between UI components.
    """

    # Signals for UI updates
    data_loaded = Signal(pd.DataFrame)
    strategy_loaded = Signal(str)  # Strategy info message
    strategies_updated = Signal(list)  # List of strategy names
    backtest_started = Signal()
    backtest_finished = Signal(list)  # List of BacktestResult
    backtest_progress = Signal(str)
    backtest_error = Signal(str)
    alert_triggered = Signal(str, str, str, str, str)  # type, severity, title, message, source

    def __init__(self):
        super().__init__()
        self.data_manager = DataManager()
        self.strategy_registry = StrategyRegistry()
        self.backtester = OptimizedBacktester()
        self.optimization_controller = OptimizationController()
        self.alert_manager = AlertManager()

        # Load available strategies
        self._load_strategies()

        # Current data and results
        self.current_data: Optional[pd.DataFrame] = None
        self.backtest_results: List[Any] = []

        # Start alert monitoring
        self.alert_manager.start_monitoring()

        # Setup default alert rules
        self._setup_default_alert_rules()

        logger.info("Dashboard controller initialized")

    def _setup_default_alert_rules(self):
        """Setup default alert rules"""
        from core.alerts import AlertRule, NotificationMethod

        # High performance alert
        high_perf_rule = AlertRule(
            id="high_performance",
            name="Alto Rendimiento",
            type=AlertType.PERFORMANCE_THRESHOLD,
            conditions={"min_severity": "medium"},
            notification_methods=[NotificationMethod.GUI, NotificationMethod.LOG]
        )
        self.alert_manager.add_rule(high_perf_rule)

        # Strategy error alert
        error_rule = AlertRule(
            id="strategy_errors",
            name="Errores de Estrategia",
            type=AlertType.STRATEGY_ERROR,
            conditions={"min_severity": "high"},
            notification_methods=[NotificationMethod.GUI, NotificationMethod.SOUND, NotificationMethod.LOG]
        )
        self.alert_manager.add_rule(error_rule)

        logger.info("Default alert rules configured")

    def _load_strategies(self):
        """Load all available strategies"""
        try:
            from core.strategies import (
                MomentumStrategy, MeanReversionStrategy, BreakoutStrategy
            )

            self.strategy_registry.register_strategy(MomentumStrategy)
            self.strategy_registry.register_strategy(MeanReversionStrategy)
            self.strategy_registry.register_strategy(BreakoutStrategy)

            strategies = self.strategy_registry.list_strategies()
            self.strategies_updated.emit(strategies)
            logger.info(f"Loaded {len(strategies)} strategies")

        except Exception as e:
            logger.error(f"Error loading strategies: {e}")

    def load_market_data(self, symbol: str = "BTC", timeframe: str = "1D",
                        days: int = 365) -> bool:
        """
        Load market data for the specified symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'BTC', 'ETH')
            timeframe: Timeframe (e.g., '1D', '1H', '15Min')
            days: Number of days of historical data

        Returns:
            True if data loaded successfully, False otherwise
        """
        try:
            self.backtest_progress.emit(f"Loading {days} days of {symbol} {timeframe} data...")

            # For demo purposes, generate synthetic data
            # In production, this would load from data_manager
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            dates = pd.date_range(start_date, end_date, freq='D')

            # Generate realistic price data
            np.random.seed(42)
            base_price = 50000 if symbol == "BTC" else 3000
            trend = np.linspace(0, base_price * 0.5, len(dates))
            noise = np.random.normal(0, base_price * 0.02, len(dates))
            cycle = base_price * 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates)))

            close_prices = base_price + trend + cycle + noise.cumsum() * 0.1

            self.current_data = pd.DataFrame({
                'open': close_prices + np.random.normal(0, base_price * 0.005, len(dates)),
                'high': close_prices + abs(np.random.normal(0, base_price * 0.01, len(dates))),
                'low': close_prices - abs(np.random.normal(0, base_price * 0.01, len(dates))),
                'close': close_prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            }, index=dates)

            # Ensure OHLC relationships
            self.current_data['high'] = self.current_data[['open', 'close', 'high']].max(axis=1)
            self.current_data['low'] = self.current_data[['open', 'close', 'low']].min(axis=1)

            self.data_loaded.emit(self.current_data)
            self.backtest_progress.emit(f"Loaded {len(self.current_data)} data points")
            logger.info(f"Market data loaded: {len(self.current_data)} points")

            return True

        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            self.backtest_error.emit(f"Failed to load data: {str(e)}")
            return False

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return self.strategy_registry.list_strategies()

    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration template for a strategy"""
        try:
            info = self.strategy_registry.get_strategy_info(strategy_name)
            return {
                'name': strategy_name,
                'description': info['docstring'][:100] if info['docstring'] else "",
                'required_parameters': info['required_parameters'],
                'default_config': self.strategy_registry.get_default_config(strategy_name)
            }
        except Exception as e:
            logger.error(f"Error getting strategy config: {e}")
            return {}

    def create_strategy_instance(self, config_dict: Dict[str, Any]) -> Any:
        """Create a strategy instance from configuration dictionary"""
        try:
            config = StrategyConfig(**config_dict)
            return self.strategy_registry.create_strategy(config)
        except Exception as e:
            logger.error(f"Error creating strategy: {e}")
            return None

    def run_backtests(self, strategy_configs: List[Dict[str, Any]]) -> bool:
        """
        Run parallel backtests for multiple strategy configurations.

        Args:
            strategy_configs: List of strategy configuration dictionaries

        Returns:
            True if backtests started successfully
        """
        if not self.current_data is not None:
            self.backtest_error.emit("No market data loaded")
            return False

        try:
            self.backtest_started.emit()
            self.backtest_progress.emit("Preparing strategies...")

            # Create strategy instances
            strategies = []
            parameter_sets = []

            for config_dict in strategy_configs:
                strategy = self.create_strategy_instance(config_dict)
                if strategy:
                    strategies.append(strategy)
                    parameter_sets.append(config_dict.get('parameters', {}))

            if not strategies:
                self.backtest_error.emit("No valid strategies created")
                return False

            # Run backtests in background thread
            self.backtest_worker = BacktestWorker(
                self.current_data, strategies, parameter_sets
            )
            self.backtest_worker.finished.connect(self._on_backtest_finished)
            self.backtest_worker.progress.connect(self.backtest_progress)
            self.backtest_worker.error.connect(self.backtest_error)
            self.backtest_worker.start()

            return True

        except Exception as e:
            logger.error(f"Error starting backtests: {e}")
            self.backtest_error.emit(f"Failed to start backtests: {str(e)}")
            return False

    def _on_backtest_finished(self, results: List[Any]):
        """Handle backtest completion"""
        self.backtest_results = results
        self.backtest_finished.emit(results)

        # Trigger alert for backtest completion
        if results:
            best_result = max(results, key=lambda r: r.metrics.get('sharpe_ratio', 0))
            best_sharpe = best_result.metrics.get('sharpe_ratio', 0)

            if best_sharpe > 2.0:
                self.trigger_alert(
                    AlertType.PERFORMANCE_THRESHOLD,
                    AlertSeverity.MEDIUM,
                    "Excelente Resultado de Backtest",
                    f"Backtest completado con Sharpe ratio de {best_sharpe:.2f}",
                    "backtester",
                    {"sharpe_ratio": best_sharpe, "total_backtests": len(results)}
                )
            elif best_sharpe > 1.0:
                self.trigger_alert(
                    AlertType.PERFORMANCE_THRESHOLD,
                    AlertSeverity.LOW,
                    "Buen Resultado de Backtest",
                    f"Backtest completado con Sharpe ratio de {best_sharpe:.2f}",
                    "backtester",
                    {"sharpe_ratio": best_sharpe, "total_backtests": len(results)}
                )

        logger.info(f"Backtests completed: {len(results)} results")

    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of backtest results"""
        if not self.backtest_results:
            return {}

        summary = {
            'total_backtests': len(self.backtest_results),
            'best_sharpe': -float('inf'),
            'best_result': None,
            'avg_execution_time': 0,
            'results': []
        }

        for result in self.backtest_results:
            metrics = result.metrics
            summary['avg_execution_time'] += result.execution_time

            result_summary = {
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'total_return': metrics.get('total_return', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'win_rate': metrics.get('win_rate', 0),
                'execution_time': result.execution_time,
                'parameter_set': result.parameter_set
            }
            summary['results'].append(result_summary)

            if metrics.get('sharpe_ratio', 0) > summary['best_sharpe']:
                summary['best_sharpe'] = metrics.get('sharpe_ratio', 0)
                summary['best_result'] = result_summary

        summary['avg_execution_time'] /= len(self.backtest_results)
        return summary

    def export_results(self, filepath: str) -> bool:
        """Export backtest results to file"""
        try:
            if not self.backtest_results:
                return False

            results_data = []
            for i, result in enumerate(self.backtest_results):
                row = {
                    'backtest_id': i,
                    'execution_time': result.execution_time,
                    **result.metrics
                }
                if result.parameter_set:
                    row.update({f'param_{k}': v for k, v in result.parameter_set.items()})
                results_data.append(row)

            df = pd.DataFrame(results_data)
            df.to_csv(filepath, index=False)
            return True

        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False

    # Optimization methods
    def setup_strategy_for_optimization(self, strategy_name: str):
        """Setup a strategy for parameter optimization"""
        try:
            strategy_class = self.strategy_registry.get_strategy_class(strategy_name)
            if strategy_class:
                self.optimization_controller.add_strategy(strategy_name, strategy_class)
                logger.info(f"Strategy {strategy_name} configured for optimization")
                return True
            else:
                logger.error(f"Strategy {strategy_name} not found in registry")
                return False
        except Exception as e:
            logger.error(f"Error setting up optimization for {strategy_name}: {e}")
            return False

    def start_strategy_optimization(self, strategy_name: str, config: Optional[Any] = None):
        """Start optimization for a strategy"""
        try:
            # Create backtest function for optimization
            def backtest_function(**params):
                return self.run_single_backtest(strategy_name, params)

            self.optimization_controller.start_optimization(strategy_name, backtest_function)
            logger.info(f"Started optimization for {strategy_name}")

        except Exception as e:
            logger.error(f"Error starting optimization for {strategy_name}: {e}")
            raise

    def cancel_strategy_optimization(self, strategy_name: str):
        """Cancel optimization for a strategy"""
        self.optimization_controller.cancel_optimization(strategy_name)

    def get_optimization_results(self, strategy_name: str) -> Optional[Dict]:
        """Get optimization results for a strategy"""
        return self.optimization_controller.get_optimization_results(strategy_name)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        return self.optimization_controller.get_optimization_summary()

    def get_available_strategies_for_optimization(self) -> List[str]:
        """Get list of available strategies for optimization"""
        return self.get_available_strategies()

    # Alert methods
    def trigger_alert(self, alert_type: AlertType, severity: AlertSeverity,
                     title: str, message: str, source: str = "system",
                     data: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Trigger an alert"""
        return self.alert_manager.trigger_alert(alert_type, severity, title, message, source, data)

    def get_alert_manager(self) -> AlertManager:
        """Get the alert manager instance"""
        return self.alert_manager

    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        self.alert_manager.acknowledge_alert(alert_id)

    def get_active_alerts(self):
        """Get active alerts"""
        return self.alert_manager.get_active_alerts()

    def get_alert_history(self, limit: Optional[int] = None):
        """Get alert history"""
        return self.alert_manager.get_alert_history(limit)

    def run_single_backtest(self, strategy_name: str, parameters: Dict[str, Any]) -> List[Any]:
        """Run a single backtest with given parameters (used by optimization)"""
        try:
            if not self.current_data is not None:
                raise ValueError("No market data loaded")

            strategy_class = self.strategy_registry.get_strategy_class(strategy_name)
            if not strategy_class:
                raise ValueError(f"Strategy {strategy_name} not found")

            # Create strategy function for backtester
            def strategy_func(data: pd.DataFrame, **params) -> pd.Series:
                strategy = strategy_class(**params)
                indicators = strategy.calculate_indicators(data)
                signals = strategy.generate_signals(indicators)
                return signals

            # Run single backtest
            results = self.backtester.run_parallel_backtests(
                self.current_data,
                strategy_func,
                [parameters]  # Single parameter set
            )

            return results

        except Exception as e:
            logger.error(f"Error running single backtest for {strategy_name}: {e}")
            # Return empty results with error
            return []