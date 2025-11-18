"""
Optimization Module - Integration with dashboard for parameter optimization
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from PySide6.QtCore import QObject, Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox, QProgressBar,
    QTextEdit, QTableWidget, QTableWidgetItem, QHeaderView, QGroupBox,
    QMessageBox
)

from .genetic_optimizer import GeneticOptimizer, OptimizationConfig, OptimizationManager

logger = logging.getLogger(__name__)

class OptimizationWorker(QThread):
    """
    Worker thread for running optimization in background
    """

    progress = Signal(str)  # Progress message
    generation_complete = Signal(int, float, dict)  # generation, best_fitness, best_params
    finished = Signal(dict)  # Final results
    error = Signal(str)  # Error message

    def __init__(self, optimizer: GeneticOptimizer, backtest_function: Callable):
        super().__init__()
        self.optimizer = optimizer
        self.backtest_function = backtest_function
        self.is_cancelled = False

    def cancel(self):
        """Cancel the optimization"""
        self.is_cancelled = True

    def run(self):
        """Run optimization in background thread"""
        try:
            # Override optimizer's progress callback
            def progress_callback(message: str):
                if self.is_cancelled:
                    raise InterruptedError("Optimization cancelled")
                self.progress.emit(message)

            # Run optimization
            results = self.optimizer.optimize(self.backtest_function, progress_callback)

            if not self.is_cancelled:
                self.finished.emit(results)

        except InterruptedError:
            self.progress.emit("Optimization cancelled")
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            self.error.emit(str(e))

class OptimizationController(QObject):
    """
    Controller for parameter optimization operations
    """

    # Signals
    optimization_started = Signal(str)  # Strategy name
    optimization_progress = Signal(str)  # Progress message
    optimization_generation = Signal(int, float, dict)  # gen, fitness, params
    optimization_finished = Signal(str, dict)  # Strategy name, results
    optimization_error = Signal(str, str)  # Strategy name, error

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = OptimizationManager()
        self.active_workers: Dict[str, OptimizationWorker] = {}
        self.optimization_configs: Dict[str, OptimizationConfig] = {}

    def add_strategy(self, strategy_name: str, strategy_class: Any,
                    config: Optional[OptimizationConfig] = None):
        """Add a strategy for optimization"""
        try:
            self.manager.add_strategy_optimization(strategy_name, strategy_class, config)
            self.optimization_configs[strategy_name] = config or OptimizationConfig()
            logger.info(f"Added strategy {strategy_name} for optimization")
        except Exception as e:
            logger.error(f"Error adding strategy {strategy_name}: {e}")
            raise

    def start_optimization(self, strategy_name: str, backtest_function: Callable):
        """Start optimization for a strategy"""
        if strategy_name in self.active_workers:
            raise ValueError(f"Optimization already running for {strategy_name}")

        if strategy_name not in self.manager.optimizers:
            raise ValueError(f"Strategy {strategy_name} not configured for optimization")

        try:
            optimizer = self.manager.optimizers[strategy_name]
            worker = OptimizationWorker(optimizer, backtest_function)

            # Connect signals
            worker.progress.connect(self._on_progress)
            worker.generation_complete.connect(self._on_generation_complete)
            worker.finished.connect(lambda results: self._on_finished(strategy_name, results))
            worker.error.connect(lambda error: self._on_error(strategy_name, error))

            # Store worker reference
            self.active_workers[strategy_name] = worker

            # Start optimization
            self.optimization_started.emit(strategy_name)
            worker.start()

            logger.info(f"Started optimization for {strategy_name}")

        except Exception as e:
            logger.error(f"Error starting optimization for {strategy_name}: {e}")
            self.optimization_error.emit(strategy_name, str(e))

    def cancel_optimization(self, strategy_name: str):
        """Cancel optimization for a strategy"""
        if strategy_name in self.active_workers:
            worker = self.active_workers[strategy_name]
            worker.cancel()
            worker.wait()  # Wait for clean shutdown
            del self.active_workers[strategy_name]
            logger.info(f"Cancelled optimization for {strategy_name}")

    def cancel_all_optimizations(self):
        """Cancel all running optimizations"""
        for strategy_name in list(self.active_workers.keys()):
            self.cancel_optimization(strategy_name)

    def get_optimization_results(self, strategy_name: str) -> Optional[Dict]:
        """Get optimization results for a strategy"""
        return self.manager.optimization_results.get(strategy_name)

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization results"""
        try:
            summary_df = self.manager.get_optimization_summary()
            return {
                'summary_table': summary_df.to_dict('records') if not summary_df.empty else [],
                'total_strategies': len(self.manager.optimizers),
                'completed_optimizations': len(self.manager.optimization_results),
                'running_optimizations': len(self.active_workers)
            }
        except Exception as e:
            logger.error(f"Error getting optimization summary: {e}")
            return {'error': str(e)}

    def is_optimization_running(self, strategy_name: str) -> bool:
        """Check if optimization is running for a strategy"""
        return strategy_name in self.active_workers

    def get_active_optimizations(self) -> List[str]:
        """Get list of strategies currently being optimized"""
        return list(self.active_workers.keys())

    def update_config(self, strategy_name: str, config: OptimizationConfig):
        """Update optimization configuration for a strategy"""
        if strategy_name in self.manager.optimizers:
            # Create new optimizer with updated config
            optimizer = GeneticOptimizer(config)
            strategy_class = self._get_strategy_class(strategy_name)
            if strategy_class:
                param_ranges = optimizer.get_parameter_ranges_from_strategy(strategy_class)
                optimizer.set_parameter_bounds(param_ranges)
                self.manager.optimizers[strategy_name] = optimizer
                self.optimization_configs[strategy_name] = config
                logger.info(f"Updated optimization config for {strategy_name}")

    def _get_strategy_class(self, strategy_name: str) -> Optional[Any]:
        """Get strategy class (placeholder - would integrate with strategy registry)"""
        # This would be implemented to get the actual strategy class
        # For now, return None
        return None

    def _on_progress(self, message: str):
        """Handle progress updates"""
        self.optimization_progress.emit(message)

    def _on_generation_complete(self, generation: int, fitness: float, params: dict):
        """Handle generation completion"""
        self.optimization_generation.emit(generation, fitness, params)

    def _on_finished(self, strategy_name: str, results: dict):
        """Handle optimization completion"""
        if strategy_name in self.active_workers:
            del self.active_workers[strategy_name]

        self.optimization_finished.emit(strategy_name, results)
        logger.info(f"Optimization completed for {strategy_name}")

    def _on_error(self, strategy_name: str, error: str):
        """Handle optimization errors"""
        if strategy_name in self.active_workers:
            del self.active_workers[strategy_name]

        self.optimization_error.emit(strategy_name, error)
        logger.error(f"Optimization error for {strategy_name}: {error}")

class OptimizationPanel(QWidget):
    """
    UI Panel for parameter optimization
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.optimization_controller = controller.optimization_controller
        self.setup_ui()

        # Connect signals
        self.optimization_controller.optimization_started.connect(self.on_optimization_started)
        self.optimization_controller.optimization_progress.connect(self.on_optimization_progress)
        self.optimization_controller.optimization_finished.connect(self.on_optimization_finished)
        self.optimization_controller.optimization_error.connect(self.on_optimization_error)

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)

        # Strategy selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QHBoxLayout(strategy_group)

        strategy_layout.addWidget(QLabel("Strategy:"))
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItem("Select Strategy...")
        strategy_layout.addWidget(self.strategy_combo)

        self.load_strategies_button = QPushButton("Load Strategies")
        self.load_strategies_button.clicked.connect(self.load_strategies)
        strategy_layout.addWidget(self.load_strategies_button)

        layout.addWidget(strategy_group)

        # Optimization configuration
        config_group = QGroupBox("Optimization Configuration")
        config_layout = QGridLayout(config_group)

        # Population size
        config_layout.addWidget(QLabel("Population Size:"), 0, 0)
        self.population_spin = QSpinBox()
        self.population_spin.setRange(10, 200)
        self.population_spin.setValue(50)
        config_layout.addWidget(self.population_spin, 0, 1)

        # Generations
        config_layout.addWidget(QLabel("Generations:"), 1, 0)
        self.generations_spin = QSpinBox()
        self.generations_spin.setRange(5, 100)
        self.generations_spin.setValue(30)
        config_layout.addWidget(self.generations_spin, 1, 1)

        # Mutation rate
        config_layout.addWidget(QLabel("Mutation Rate:"), 2, 0)
        self.mutation_spin = QDoubleSpinBox()
        self.mutation_spin.setRange(0.01, 0.5)
        self.mutation_spin.setSingleStep(0.01)
        self.mutation_spin.setValue(0.1)
        config_layout.addWidget(self.mutation_spin, 2, 1)

        # Fitness function
        config_layout.addWidget(QLabel("Fitness Function:"), 3, 0)
        self.fitness_combo = QComboBox()
        self.fitness_combo.addItems([
            "sharpe_ratio", "total_return", "calmar_ratio", "sortino_ratio"
        ])
        config_layout.addWidget(self.fitness_combo, 3, 1)

        layout.addWidget(config_group)

        # Control buttons
        controls_group = QGroupBox("Optimization Controls")
        controls_layout = QHBoxLayout(controls_group)

        self.start_button = QPushButton("Start Optimization")
        self.start_button.clicked.connect(self.start_optimization)
        self.start_button.setEnabled(False)
        controls_layout.addWidget(self.start_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_optimization)
        self.cancel_button.setEnabled(False)
        controls_layout.addWidget(self.cancel_button)

        controls_layout.addStretch()
        layout.addWidget(controls_group)

        # Progress and status
        progress_group = QGroupBox("Progress")
        progress_layout = QVBoxLayout(progress_group)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)

        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(100)
        self.status_text.setReadOnly(True)
        progress_layout.addWidget(self.status_text)

        layout.addWidget(progress_group)

        # Results display
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Strategy", "Best Fitness", "Sharpe Ratio", "Total Return"
        ])
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.results_table.setAlternatingRowColors(True)

        results_layout.addWidget(self.results_table)
        layout.addWidget(results_group)

        # Best parameters display
        params_group = QGroupBox("Best Parameters")
        params_layout = QVBoxLayout(params_group)

        self.params_text = QTextEdit()
        self.params_text.setReadOnly(True)
        self.params_text.setMaximumHeight(150)

        params_layout.addWidget(self.params_text)
        layout.addWidget(params_group)

    def load_strategies(self):
        """Load available strategies for optimization"""
        try:
            # Get strategies from controller
            strategies = self.controller.get_available_strategies()

            self.strategy_combo.clear()
            self.strategy_combo.addItem("Select Strategy...")

            for strategy_name in strategies:
                self.strategy_combo.addItem(strategy_name)

            if strategies:
                self.start_button.setEnabled(True)
                self.update_status(f"Loaded {len(strategies)} strategies for optimization")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load strategies: {str(e)}")

    def start_optimization(self):
        """Start optimization for selected strategy"""
        strategy_name = self.strategy_combo.currentText()

        if strategy_name == "Select Strategy...":
            QMessageBox.warning(self, "Warning", "Please select a strategy")
            return

        try:
            # Create optimization config
            config = OptimizationConfig(
                population_size=self.population_spin.value(),
                generations=self.generations_spin.value(),
                mutation_rate=self.mutation_spin.value(),
                fitness_function=self.fitness_combo.currentText()
            )

            # Update config
            self.optimization_controller.update_config(strategy_name, config)

            # Create backtest function
            backtest_func = self.create_backtest_function(strategy_name)

            # Start optimization
            self.optimization_controller.start_optimization(strategy_name, backtest_func)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization: {str(e)}")

    def cancel_optimization(self):
        """Cancel current optimization"""
        strategy_name = self.strategy_combo.currentText()
        if strategy_name != "Select Strategy...":
            self.optimization_controller.cancel_optimization(strategy_name)

    def create_backtest_function(self, strategy_name: str) -> Callable:
        """Create backtest function for optimization"""
        def backtest_function(**params):
            # This would integrate with the backtesting system
            # For now, return mock results
            return self.controller.run_single_backtest(strategy_name, params)

        return backtest_function

    def on_optimization_started(self, strategy_name: str):
        """Handle optimization start"""
        self.progress_bar.setVisible(True)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.update_status(f"Started optimization for {strategy_name}")

    def on_optimization_progress(self, message: str):
        """Handle optimization progress"""
        self.update_status(message)

    def on_optimization_finished(self, strategy_name: str, results: dict):
        """Handle optimization completion"""
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        self.update_status(f"Optimization completed for {strategy_name}")
        self.display_results(strategy_name, results)

    def on_optimization_error(self, strategy_name: str, error: str):
        """Handle optimization error"""
        self.progress_bar.setVisible(False)
        self.start_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

        self.update_status(f"Error in optimization for {strategy_name}: {error}")
        QMessageBox.critical(self, "Optimization Error", f"{strategy_name}: {error}")

    def update_status(self, message: str):
        """Update status display"""
        self.status_text.setPlainText(message)
        logger.info(f"Optimization status: {message}")

    def display_results(self, strategy_name: str, results: dict):
        """Display optimization results"""
        # Update results table
        row_count = self.results_table.rowCount()
        self.results_table.insertRow(row_count)

        self.results_table.setItem(row_count, 0, QTableWidgetItem(strategy_name))
        self.results_table.setItem(row_count, 1, QTableWidgetItem(f"{results['best_fitness']:.4f}"))
        self.results_table.setItem(row_count, 2, QTableWidgetItem(f"{results['best_metrics']['sharpe_ratio']:.3f}"))
        self.results_table.setItem(row_count, 3, QTableWidgetItem(f"{results['best_metrics']['total_return']:.2%}"))

        # Display best parameters
        params_text = f"Best Parameters for {strategy_name}:\n\n"
        for param, value in results['best_parameters'].items():
            params_text += f"{param}: {value}\n"

        params_text += f"\nBest Fitness: {results['best_fitness']:.4f}\n"
        params_text += f"Optimization Time: {results['optimization_time']:.2f}s\n"
        params_text += f"Convergence Generation: {results['convergence_generation']}"

        self.params_text.setPlainText(params_text)

        self.results_table.resizeColumnsToContents()