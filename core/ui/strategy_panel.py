"""
Strategy Panel - UI for strategy selection and configuration
"""

import json
import logging
from typing import Dict, List, Any, Optional
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTextEdit, QListWidget, QListWidgetItem, QMessageBox,
    QSplitter, QFrame
)
from PySide6.QtCore import Qt, Signal

logger = logging.getLogger(__name__)

class StrategyPanel(QWidget):
    """
    Panel for strategy selection, configuration, and management.
    """

    strategy_selected = Signal(str)  # Emitted when a strategy is selected
    strategy_configured = Signal(dict)  # Emitted when strategy is configured

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.current_strategy = None
        self.strategy_configs = []
        self.setup_ui()

        # Connect to controller signals
        self.controller.strategies_updated.connect(self.update_strategy_list)

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)  # Add margins
        layout.setSpacing(15)  # Increase spacing between groups

        # Strategy selection
        strategy_group = QGroupBox("📈 Strategy Selection")
        strategy_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        strategy_layout = QVBoxLayout(strategy_group)
        strategy_layout.setContentsMargins(15, 20, 15, 15)  # More padding inside groups
        strategy_layout.setSpacing(10)

        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumHeight(30)  # Larger combo box
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_changed)
        strategy_layout.addWidget(QLabel("Available Strategies:"))
        strategy_layout.addWidget(self.strategy_combo)

        # Strategy description
        self.strategy_desc = QTextEdit()
        self.strategy_desc.setMaximumHeight(80)
        self.strategy_desc.setReadOnly(True)
        self.strategy_desc.setStyleSheet("QTextEdit { background-color: #f8f9fa; border: 1px solid #dee2e6; }")
        strategy_layout.addWidget(QLabel("Description:"))
        strategy_layout.addWidget(self.strategy_desc)

        layout.addWidget(strategy_group)

        # Parameter configuration
        params_group = QGroupBox("⚙️ Parameters")
        params_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        self.params_layout = QVBoxLayout(params_group)
        self.params_layout.setContentsMargins(15, 20, 15, 15)

        # Scroll area for parameters
        self.params_frame = QFrame()
        self.params_form_layout = QFormLayout(self.params_frame)
        self.params_form_layout.setVerticalSpacing(12)  # More spacing between form rows
        self.params_form_layout.setHorizontalSpacing(15)
        self.params_layout.addWidget(self.params_frame)

        layout.addWidget(params_group)

        # Risk management
        risk_group = QGroupBox("🛡️ Risk Management")
        risk_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        risk_layout = QFormLayout(risk_group)
        risk_layout.setContentsMargins(15, 20, 15, 15)
        risk_layout.setVerticalSpacing(12)
        risk_layout.setHorizontalSpacing(15)

        self.max_dd_spin = QDoubleSpinBox()
        self.max_dd_spin.setRange(0.01, 1.0)
        self.max_dd_spin.setValue(0.15)
        self.max_dd_spin.setSingleStep(0.01)
        self.max_dd_spin.setMinimumHeight(25)
        risk_layout.addRow("Max Drawdown Threshold:", self.max_dd_spin)

        self.volatility_spin = QDoubleSpinBox()
        self.volatility_spin.setRange(0.001, 0.1)
        self.volatility_spin.setValue(0.03)
        self.volatility_spin.setSingleStep(0.005)
        self.volatility_spin.setMinimumHeight(25)
        risk_layout.addRow("Volatility Threshold:", self.volatility_spin)

        layout.addWidget(risk_group)

        # Filters
        filters_group = QGroupBox("🔍 Filters")
        filters_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        filters_layout = QVBoxLayout(filters_group)
        filters_layout.setContentsMargins(15, 20, 15, 15)
        filters_layout.setSpacing(8)

        self.dd_filter_check = QCheckBox("Max Drawdown Filter")
        self.dd_filter_check.setChecked(True)
        filters_layout.addWidget(self.dd_filter_check)

        self.volatility_filter_check = QCheckBox("Volatility Filter")
        self.volatility_filter_check.setChecked(True)
        filters_layout.addWidget(self.volatility_filter_check)

        self.trend_filter_check = QCheckBox("Trend Filter")
        self.trend_filter_check.setChecked(True)
        filters_layout.addWidget(self.trend_filter_check)

        layout.addWidget(filters_group)

        # Action buttons
        buttons_group = QGroupBox("Actions")
        buttons_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        buttons_layout = QHBoxLayout(buttons_group)
        buttons_layout.setContentsMargins(15, 20, 15, 15)
        buttons_layout.setSpacing(15)

        self.add_button = QPushButton("➕ Add")
        self.add_button.setMaximumHeight(26)
        self.add_button.setMaximumWidth(80)
        self.add_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 4px 8px;
                background-color: #007bff;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #0056b3;
            }
            QPushButton:pressed {
                background-color: #004085;
            }
        """)
        self.add_button.clicked.connect(self.add_strategy_config)
        buttons_layout.addWidget(self.add_button)

        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.setMaximumHeight(26)
        self.clear_button.setMaximumWidth(70)
        self.clear_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 4px 8px;
                background-color: #dc3545;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        self.clear_button.clicked.connect(self.clear_configs)
        buttons_layout.addWidget(self.clear_button)

        layout.addWidget(buttons_group)

        # Configuration list
        config_group = QGroupBox("📋 Backtest Configurations")
        config_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                font-size: 12px;
                padding-top: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        config_layout = QVBoxLayout(config_group)
        config_layout.setContentsMargins(15, 20, 15, 15)

        self.config_list = QListWidget()
        self.config_list.setMinimumHeight(120)
        self.config_list.setStyleSheet("""
            QListWidget {
                border: 1px solid #dee2e6;
                border-radius: 5px;
                background-color: #f8f9fa;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
            }
            QListWidget::item:selected {
                background-color: #007bff;
                color: white;
            }
        """)
        self.config_list.itemDoubleClicked.connect(self.edit_config)
        config_layout.addWidget(self.config_list)

        # Config action buttons
        config_buttons = QHBoxLayout()
        config_buttons.setSpacing(15)

        self.remove_config_button = QPushButton("❌ Remove")
        self.remove_config_button.setMaximumHeight(24)
        self.remove_config_button.setMaximumWidth(85)
        self.remove_config_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 3px 8px;
                background-color: #ffc107;
                color: black;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #e0a800;
            }
        """)
        self.remove_config_button.clicked.connect(self.remove_selected_config)
        config_buttons.addWidget(self.remove_config_button)

        self.edit_config_button = QPushButton("✏️ Edit")
        self.edit_config_button.setMaximumHeight(24)
        self.edit_config_button.setMaximumWidth(70)
        self.edit_config_button.setStyleSheet("""
            QPushButton {
                font-size: 9px;
                padding: 3px 8px;
                background-color: #28a745;
                color: white;
                border: none;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #218838;
            }
        """)
        self.edit_config_button.clicked.connect(self.edit_selected_config)
        config_buttons.addWidget(self.edit_config_button)

        config_layout.addLayout(config_buttons)
        layout.addWidget(config_group)

        # Initialize parameter widgets storage
        self.param_widgets = {}

    def update_strategy_list(self, strategies: List[str]):
        """Update the list of available strategies"""
        self.strategy_combo.clear()
        self.strategy_combo.addItems(strategies)

        if strategies:
            self.on_strategy_changed(strategies[0])

    def on_strategy_changed(self, strategy_name: str):
        """Handle strategy selection change"""
        if not strategy_name:
            return

        self.current_strategy = strategy_name
        config = self.controller.get_strategy_config(strategy_name)

        # Update description
        self.strategy_desc.setPlainText(config.get('description', ''))

        # Clear existing parameter widgets
        self.clear_parameter_widgets()

        # Create parameter widgets
        required_params = config.get('required_parameters', [])
        for param in required_params:
            self.create_parameter_widget(param)

        self.strategy_selected.emit(strategy_name)

    def create_parameter_widget(self, param_name: str):
        """Create a widget for a parameter"""
        if param_name == 'fast_period':
            widget = QSpinBox()
            widget.setRange(1, 100)
            widget.setValue(10)
        elif param_name == 'slow_period':
            widget = QSpinBox()
            widget.setRange(1, 200)
            widget.setValue(20)
        elif param_name == 'momentum_threshold':
            widget = QDoubleSpinBox()
            widget.setRange(0.001, 0.5)
            widget.setValue(0.02)
            widget.setSingleStep(0.005)
        elif param_name == 'lookback_period':
            widget = QSpinBox()
            widget.setRange(5, 200)
            widget.setValue(20)
        elif param_name == 'entry_threshold':
            widget = QDoubleSpinBox()
            widget.setRange(0.1, 5.0)
            widget.setValue(2.0)
            widget.setSingleStep(0.1)
        elif param_name == 'exit_threshold':
            widget = QDoubleSpinBox()
            widget.setRange(0.01, 2.0)
            widget.setValue(0.5)
            widget.setSingleStep(0.05)
        elif param_name == 'breakout_threshold':
            widget = QDoubleSpinBox()
            widget.setRange(0.001, 0.2)
            widget.setValue(0.05)
            widget.setSingleStep(0.005)
        elif param_name == 'consolidation_period':
            widget = QSpinBox()
            widget.setRange(5, 100)
            widget.setValue(10)
        else:
            # Default to double spin box
            widget = QDoubleSpinBox()
            widget.setRange(0, 1000)
            widget.setValue(1.0)

        self.param_widgets[param_name] = widget
        self.params_form_layout.addRow(f"{param_name}:", widget)

    def clear_parameter_widgets(self):
        """Clear all parameter widgets"""
        # Remove all rows from form layout
        while self.params_form_layout.rowCount() > 0:
            self.params_form_layout.removeRow(0)

        self.param_widgets.clear()

    def get_current_config(self) -> Dict[str, Any]:
        """Get the current strategy configuration"""
        if not self.current_strategy:
            return {}

        # Get parameters
        parameters = {}
        for param_name, widget in self.param_widgets.items():
            if isinstance(widget, QSpinBox):
                parameters[param_name] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                parameters[param_name] = widget.value()

        # Get risk management
        risk_management = {
            'max_drawdown_threshold': self.max_dd_spin.value(),
            'volatility_threshold': self.volatility_spin.value()
        }

        # Get filters
        filters = []
        if self.dd_filter_check.isChecked():
            filters.append('max_drawdown_filter')
        if self.volatility_filter_check.isChecked():
            filters.append('volatility_filter')
        if self.trend_filter_check.isChecked():
            filters.append('trend_filter')

        return {
            'name': self.current_strategy,
            'description': f"Configured {self.current_strategy}",
            'parameters': parameters,
            'risk_management': risk_management,
            'filters': filters
        }

    def add_strategy_config(self):
        """Add current configuration to backtest list"""
        config = self.get_current_config()
        if not config:
            QMessageBox.warning(self, "Warning", "No strategy selected")
            return

        # Check if config already exists
        config_str = json.dumps(config, sort_keys=True)
        for existing_config in self.strategy_configs:
            existing_str = json.dumps(existing_config, sort_keys=True)
            if config_str == existing_str:
                QMessageBox.information(self, "Info", "This configuration already exists")
                return

        self.strategy_configs.append(config)
        self.update_config_list()
        self.strategy_configured.emit(config)

        logger.info(f"Added strategy config: {config['name']}")

    def update_config_list(self):
        """Update the configuration list display"""
        self.config_list.clear()
        for i, config in enumerate(self.strategy_configs):
            item_text = f"{i+1}. {config['name']} - {config['parameters']}"
            item = QListWidgetItem(item_text)
            item.setData(Qt.ItemDataRole.UserRole, i)
            self.config_list.addItem(item)

    def remove_selected_config(self):
        """Remove the selected configuration"""
        current_item = self.config_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "No configuration selected")
            return

        index = current_item.data(Qt.ItemDataRole.UserRole)
        if 0 <= index < len(self.strategy_configs):
            del self.strategy_configs[index]
            self.update_config_list()

    def edit_selected_config(self):
        """Edit the selected configuration"""
        current_item = self.config_list.currentItem()
        if not current_item:
            QMessageBox.warning(self, "Warning", "No configuration selected")
            return

        index = current_item.data(Qt.ItemDataRole.UserRole)
        if 0 <= index < len(self.strategy_configs):
            config = self.strategy_configs[index]
            self.load_config(config)

    def edit_config(self, item: QListWidgetItem):
        """Handle double-click on config item"""
        index = item.data(Qt.ItemDataRole.UserRole)
        if 0 <= index < len(self.strategy_configs):
            config = self.strategy_configs[index]
            self.load_config(config)

    def load_config(self, config: Dict[str, Any]):
        """Load a configuration into the UI"""
        # Set strategy
        strategy_name = config['name']
        index = self.strategy_combo.findText(strategy_name)
        if index >= 0:
            self.strategy_combo.setCurrentIndex(index)

        # Set parameters
        parameters = config.get('parameters', {})
        for param_name, value in parameters.items():
            if param_name in self.param_widgets:
                widget = self.param_widgets[param_name]
                if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(value)

        # Set risk management
        risk_mgmt = config.get('risk_management', {})
        self.max_dd_spin.setValue(risk_mgmt.get('max_drawdown_threshold', 0.15))
        self.volatility_spin.setValue(risk_mgmt.get('volatility_threshold', 0.03))

        # Set filters
        filters = config.get('filters', [])
        self.dd_filter_check.setChecked('max_drawdown_filter' in filters)
        self.volatility_filter_check.setChecked('volatility_filter' in filters)
        self.trend_filter_check.setChecked('trend_filter' in filters)

    def clear_configs(self):
        """Clear all configurations"""
        self.strategy_configs.clear()
        self.update_config_list()

    def get_all_configs(self) -> List[Dict[str, Any]]:
        """Get all configured strategies for backtesting"""
        return self.strategy_configs.copy()

    def set_configs(self, configs: List[Dict[str, Any]]):
        """Set the list of configurations"""
        self.strategy_configs = configs.copy()
        self.update_config_list()