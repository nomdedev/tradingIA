from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QSlider, QGroupBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QLineEdit, QHeaderView, QFrame, QScrollArea,
    QButtonGroup, QRadioButton, QSpinBox, QMessageBox, QSplitter
)
from PySide6.QtCore import Qt, Signal, QTimer, QThread
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtGui import QColor
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

# Import strategy loader
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from strategies import list_available_strategies, load_strategy


class SignalPreviewThread(QThread):
    """Background thread for generating signal preview"""
    preview_ready = Signal(object)
    
    def __init__(self, strategy, data, params):
        super().__init__()
        self.strategy = strategy
        self.data = data
        self.params = params
        
    def run(self):
        try:
            # Generate signals
            signals = self.strategy.generate_signals(self.data)
            self.preview_ready.emit(signals)
        except Exception as e:
            logging.error(f"Error in signal preview thread: {e}")
            self.preview_ready.emit(None)


class Tab2StrategyConfig(QWidget):
    """
    Strategy Configuration Tab with TWO MODES:
    
    1. PRODUCTION MODE: Use saved strategies for live trading on Alpaca
       - Load pre-configured strategies
       - Connect to Alpaca API
       - Real-time validation
       - Deploy to live trading
    
    2. RESEARCH MODE: Create and test new strategies with historical data
       - Configure new strategy parameters
       - Test on historical BTC/crypto data
       - Real-time signal preview
       - Save for backtesting
    """
    config_ready = Signal(dict)
    status_update = Signal(str, str)  # message, type (success/error/warning/info)

    def __init__(self, parent_platform, backend):
        super().__init__()
        self.parent_platform = parent_platform
        self.backend = backend
        self.current_strategy = None
        self.param_widgets = {}
        self.logger = logging.getLogger(__name__)
        self.preview_thread = None
        self.current_mode = "research"  # Default to research mode
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(12)
        
        # Mode Selector at the top
        mode_section = self.create_mode_selector()
        main_layout.addWidget(mode_section)
        
        # Main content area with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Configuration
        left_panel = self.create_configuration_panel()
        splitter.addWidget(left_panel)
        
        # Right panel: Preview
        right_panel = self.create_preview_panel()
        splitter.addWidget(right_panel)
        
        # Set initial splitter sizes (60% config, 40% preview)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter, 1)
        
        # Bottom action bar
        action_bar = self.create_action_bar()
        main_layout.addWidget(action_bar)
        
        self.setLayout(main_layout)
        
        # Load initial content
        self.on_mode_changed()
        
    def create_mode_selector(self):
        """Create mode selection section"""
        frame = QFrame()
        frame.setObjectName("modeFrame")
        frame.setStyleSheet("""
            QFrame#modeFrame {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                padding: 12px;
            }
            QRadioButton {
                font-size: 13px;
                font-weight: 500;
                padding: 8px 16px;
                spacing: 8px;
            }
            QRadioButton::indicator {
                width: 18px;
                height: 18px;
            }
            QRadioButton::indicator::checked {
                background-color: #0e639c;
                border: 2px solid #0e639c;
                border-radius: 9px;
            }
            QRadioButton::indicator::unchecked {
                background-color: #1e1e1e;
                border: 2px solid #3e3e3e;
                border-radius: 9px;
            }
        """)
        
        layout = QVBoxLayout()
        
        # Title
        title = QLabel("⚙️ Strategy Mode")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Radio buttons
        radio_layout = QHBoxLayout()
        
        self.mode_group = QButtonGroup()
        
        self.research_radio = QRadioButton("🔬 Research Mode")
        self.research_radio.setToolTip("Create and test new strategies with historical data (BTC, ETH, etc.)")
        self.research_radio.setChecked(True)
        self.mode_group.addButton(self.research_radio, 0)
        
        self.production_radio = QRadioButton("🚀 Production Mode")
        self.production_radio.setToolTip("Use saved strategies for live trading on Alpaca")
        self.mode_group.addButton(self.production_radio, 1)
        
        radio_layout.addWidget(self.research_radio)
        radio_layout.addWidget(self.production_radio)
        radio_layout.addStretch()
        
        # Mode description
        self.mode_description = QLabel()
        self.mode_description.setWordWrap(True)
        self.mode_description.setStyleSheet("color: #cccccc; font-size: 12px; padding: 8px;")
        
        layout.addLayout(radio_layout)
        layout.addWidget(self.mode_description)
        
        frame.setLayout(layout)
        
        # Connect signals
        self.mode_group.buttonClicked.connect(self.on_mode_changed)
        
        return frame
        
    def create_configuration_panel(self):
        """Create left configuration panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Strategy Selection Section
        self.strategy_group = QGroupBox("📊 Strategy Selection")
        self.strategy_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        """)
        strategy_layout = QVBoxLayout()
        
        # Strategy dropdown
        self.strategy_combo = QComboBox()
        self.strategy_combo.setMinimumHeight(32)
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_selected)
        
        # Strategy description
        self.strategy_desc = QTextEdit()
        self.strategy_desc.setMaximumHeight(80)
        self.strategy_desc.setReadOnly(True)
        self.strategy_desc.setPlaceholderText("Select a strategy to view description...")
        
        strategy_layout.addWidget(QLabel("Strategy:"))
        strategy_layout.addWidget(self.strategy_combo)
        strategy_layout.addWidget(QLabel("Description:"))
        strategy_layout.addWidget(self.strategy_desc)
        self.strategy_group.setLayout(strategy_layout)
        layout.addWidget(self.strategy_group)
        
        # Parameters Section (scrollable)
        params_group = QGroupBox("🎛️ Parameters")
        params_group.setStyleSheet(self.strategy_group.styleSheet())
        params_outer_layout = QVBoxLayout()
        
        # Scroll area for parameters
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setMinimumHeight(200)
        
        params_widget = QWidget()
        self.params_layout = QVBoxLayout()
        self.params_layout.addWidget(QLabel("No strategy selected"))
        params_widget.setLayout(self.params_layout)
        scroll.setWidget(params_widget)
        
        params_outer_layout.addWidget(scroll)
        params_group.setLayout(params_outer_layout)
        layout.addWidget(params_group, 1)
        
        # Preset Management Section (only in research mode)
        self.preset_group = QGroupBox("💾 Presets")
        self.preset_group.setStyleSheet(self.strategy_group.styleSheet())
        preset_layout = QHBoxLayout()
        
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Preset name...")
        self.preset_name_edit.setMaximumWidth(150)
        
        self.save_preset_btn = QPushButton("💾 Save")
        self.save_preset_btn.clicked.connect(self.on_save_preset)
        self.save_preset_btn.setEnabled(False)
        
        self.preset_combo = QComboBox()
        self.preset_combo.setMaximumWidth(180)
        
        self.load_preset_btn = QPushButton("📂 Load")
        self.load_preset_btn.clicked.connect(self.on_load_preset)
        self.load_preset_btn.setEnabled(False)
        
        preset_layout.addWidget(self.preset_name_edit)
        preset_layout.addWidget(self.save_preset_btn)
        preset_layout.addStretch()
        preset_layout.addWidget(self.preset_combo)
        preset_layout.addWidget(self.load_preset_btn)
        self.preset_group.setLayout(preset_layout)
        layout.addWidget(self.preset_group)
        
        # Asset Selection (for research mode)
        self.asset_group = QGroupBox("📈 Test Asset")
        self.asset_group.setStyleSheet(self.strategy_group.styleSheet())
        asset_layout = QHBoxLayout()
        
        self.asset_combo = QComboBox()
        self.asset_combo.addItems(["BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "ADA/USD"])
        self.asset_combo.currentTextChanged.connect(self.on_asset_changed)
        
        self.timeframe_combo = QComboBox()
        self.timeframe_combo.addItems(["1min", "5min", "15min", "1hour", "4hour", "1day"])
        self.timeframe_combo.setCurrentText("5min")
        self.timeframe_combo.currentTextChanged.connect(self.on_timeframe_changed)
        
        asset_layout.addWidget(QLabel("Asset:"))
        asset_layout.addWidget(self.asset_combo, 1)
        asset_layout.addWidget(QLabel("Timeframe:"))
        asset_layout.addWidget(self.timeframe_combo, 1)
        self.asset_group.setLayout(asset_layout)
        layout.addWidget(self.asset_group)
        
        # Production mode: Saved strategies list
        self.saved_strategies_group = QGroupBox("💼 Saved Strategies")
        self.saved_strategies_group.setStyleSheet(self.strategy_group.styleSheet())
        saved_layout = QVBoxLayout()
        
        self.saved_strategies_table = QTableWidget()
        self.saved_strategies_table.setColumnCount(4)
        self.saved_strategies_table.setHorizontalHeaderLabels(["Name", "Strategy", "Created", "Performance"])
        self.saved_strategies_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.saved_strategies_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.saved_strategies_table.setMaximumHeight(250)
        self.saved_strategies_table.itemSelectionChanged.connect(self.on_saved_strategy_selected)
        
        saved_layout.addWidget(self.saved_strategies_table)
        self.saved_strategies_group.setLayout(saved_layout)
        layout.addWidget(self.saved_strategies_group)
        
        widget.setLayout(layout)
        return widget
        
    def create_preview_panel(self):
        """Create right preview panel"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Validation Status Banner
        self.validation_banner = QFrame()
        self.validation_banner.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-left: 4px solid #569cd6;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        banner_layout = QHBoxLayout()
        self.validation_icon = QLabel("ℹ️")
        self.validation_text = QLabel("Configure strategy parameters to see validation")
        self.validation_text.setWordWrap(True)
        banner_layout.addWidget(self.validation_icon)
        banner_layout.addWidget(self.validation_text, 1)
        self.validation_banner.setLayout(banner_layout)
        layout.addWidget(self.validation_banner)
        
        # Signal Preview Chart
        chart_group = QGroupBox("📊 Signal Preview")
        chart_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        chart_layout = QVBoxLayout()
        
        # Chart controls
        controls_layout = QHBoxLayout()
        
        self.preview_bars_spin = QSpinBox()
        self.preview_bars_spin.setMaximumWidth(120)  # Ancho optimizado para valores con sufijo
        self.preview_bars_spin.setRange(50, 500)
        self.preview_bars_spin.setValue(100)
        self.preview_bars_spin.setSuffix(" bars")
        self.preview_bars_spin.valueChanged.connect(self.update_signal_preview)
        
        self.refresh_preview_btn = QPushButton("🔄 Refresh")
        self.refresh_preview_btn.clicked.connect(self.update_signal_preview)
        
        controls_layout.addWidget(QLabel("Preview length:"))
        controls_layout.addWidget(self.preview_bars_spin)
        controls_layout.addStretch()
        controls_layout.addWidget(self.refresh_preview_btn)
        
        chart_layout.addLayout(controls_layout)
        
        # Web view for Plotly chart
        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(300)
        chart_layout.addWidget(self.chart_view, 1)
        
        chart_group.setLayout(chart_layout)
        layout.addWidget(chart_group, 1)
        
        # Signal Statistics
        stats_group = QGroupBox("📈 Signal Statistics")
        stats_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        stats_layout = QVBoxLayout()
        
        # Stats grid
        stats_grid = QHBoxLayout()
        
        self.total_signals_label = self.create_stat_widget("Total Signals", "0")
        self.buy_signals_label = self.create_stat_widget("Buy Signals", "0", "#4ec9b0")
        self.sell_signals_label = self.create_stat_widget("Sell Signals", "0", "#f48771")
        self.signal_rate_label = self.create_stat_widget("Signal Rate", "0%")
        
        stats_grid.addWidget(self.total_signals_label)
        stats_grid.addWidget(self.buy_signals_label)
        stats_grid.addWidget(self.sell_signals_label)
        stats_grid.addWidget(self.signal_rate_label)
        
        stats_layout.addLayout(stats_grid)
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        widget.setLayout(layout)
        return widget
        
    def create_stat_widget(self, title, value, color="#569cd6"):
        """Create a small stat display widget"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border-left: 3px solid {color};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #cccccc; font-size: 11px;")
        
        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        frame.setLayout(layout)
        
        return frame
        
    def create_action_bar(self):
        """Create bottom action bar"""
        frame = QFrame()
        frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 6px;
                padding: 8px;
            }
        """)
        layout = QHBoxLayout()
        
        # Status indicator
        self.action_status = QLabel("Ready")
        self.action_status.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        
        layout.addWidget(self.action_status)
        layout.addStretch()
        
        # Action buttons (change based on mode)
        self.validate_btn = QPushButton("✓ Validate Strategy")
        self.validate_btn.clicked.connect(self.validate_strategy)
        self.validate_btn.setMinimumHeight(36)
        self.validate_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        
        self.action_btn = QPushButton("💾 Save for Backtesting")
        self.action_btn.clicked.connect(self.on_action_button_clicked)
        self.action_btn.setMinimumHeight(36)
        self.action_btn.setStyleSheet("""
            QPushButton {
                background-color: #4ec9b0;
                color: #1e1e1e;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #6fdfcf;
            }
        """)
        
        layout.addWidget(self.validate_btn)
        layout.addWidget(self.action_btn)
        
        frame.setLayout(layout)
        return frame
        
    def on_mode_changed(self):
        """Handle mode change between Research and Production"""
        if self.research_radio.isChecked():
            self.current_mode = "research"
            self.mode_description.setText(
                "🔬 Research Mode: Create and test new strategies with historical crypto data. "
                "Configure parameters, preview signals, and save configurations for backtesting."
            )
            
            # Show research mode widgets
            self.strategy_group.setVisible(True)
            self.preset_group.setVisible(True)
            self.asset_group.setVisible(True)
            self.saved_strategies_group.setVisible(False)
            
            # Update action button
            self.action_btn.setText("💾 Save for Backtesting")
            
            # Load available strategies
            self.populate_strategies()
            
        else:
            self.current_mode = "production"
            self.mode_description.setText(
                "🚀 Production Mode: Deploy tested strategies to live trading on Alpaca. "
                "Load saved strategy configurations and connect to your Alpaca account."
            )
            
            # Show production mode widgets
            self.strategy_group.setVisible(False)
            self.preset_group.setVisible(False)
            self.asset_group.setVisible(False)
            self.saved_strategies_group.setVisible(True)
            
            # Update action button
            self.action_btn.setText("🚀 Deploy to Live Trading")
            
            # Load saved strategies
            self.load_saved_strategies()
            
        self.status_update.emit(f"Switched to {self.current_mode.title()} Mode", "info")
        
    def populate_strategies(self):
        """Load available strategy modules from presets folder"""
        try:
            strategies = list_available_strategies()
            self.strategy_combo.clear()
            
            if strategies:
                self.strategy_combo.addItems(strategies)
            else:
                self.strategy_combo.addItem("No strategies available")
                
        except Exception as e:
            self.logger.error(f"Error populating strategies: {e}")
            self.strategy_combo.addItem("Error loading strategies")
            
    def load_saved_strategies(self):
        """Load saved production strategies from config"""
        try:
            self.saved_strategies_table.setRowCount(0)
            
            # Look for saved strategy configs
            config_dir = "config/strategies/production"
            if not os.path.exists(config_dir):
                os.makedirs(config_dir, exist_ok=True)
                return
                
            for filename in os.listdir(config_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(config_dir, filename)
                    with open(filepath, 'r') as f:
                        config = json.load(f)
                        
                    row = self.saved_strategies_table.rowCount()
                    self.saved_strategies_table.insertRow(row)
                    
                    self.saved_strategies_table.setItem(row, 0, QTableWidgetItem(config.get('name', 'Unknown')))
                    self.saved_strategies_table.setItem(row, 1, QTableWidgetItem(config.get('strategy', 'Unknown')))
                    self.saved_strategies_table.setItem(row, 2, QTableWidgetItem(config.get('created', 'N/A')))
                    self.saved_strategies_table.setItem(row, 3, QTableWidgetItem(config.get('performance', 'N/A')))
                    
        except Exception as e:
            self.logger.error(f"Error loading saved strategies: {e}")
            
    def on_saved_strategy_selected(self):
        """Handle selection of saved strategy in production mode"""
        selected_items = self.saved_strategies_table.selectedItems()
        if not selected_items:
            return
            
        row = selected_items[0].row()
        strategy_name = self.saved_strategies_table.item(row, 0).text()
        
        self.action_status.setText(f"Selected: {strategy_name}")
        self.action_status.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        
    def on_strategy_selected(self, strategy_name):
        """Handle strategy selection"""
        if not strategy_name or "Error" in strategy_name or "No strategies" in strategy_name:
            return
            
        try:
            # Load strategy using strategy loader
            strategy_class = load_strategy(strategy_name)
            if not strategy_class:
                self.status_update.emit(f"Failed to load strategy: {strategy_name}", "error")
                return
                
            self.current_strategy = strategy_class
            
            # Get strategy description from class docstring or predefined
            descriptions = {
                "RSI Mean Reversion": "RSI-based mean reversion strategy. Uses RSI oscillator to identify overbought/oversold conditions for mean reversion signals.",
                "MACD Momentum": "MACD momentum strategy. Combines MACD indicator with signal line crossovers for momentum-based trading signals.",
                "Bollinger Bands": "Bollinger Bands breakout strategy. Uses band touches and breakouts for volatility-based trading signals.",
                "MA Crossover": "Moving Average crossover strategy. Uses dual moving average crossovers for trend-following signals.",
                "Volume Breakout": "Volume-confirmed breakout strategy. Combines price breakouts with volume confirmation for high-probability signals."
            }
            
            self.strategy_desc.setPlainText(descriptions.get(strategy_name, strategy_class.__doc__ or "No description available."))
            
            # Clear and rebuild parameter widgets
            self.clear_param_widgets()
            
            # Get parameters from strategy class
            if hasattr(strategy_class, 'get_default_params'):
                params = strategy_class.get_default_params()
                self.create_param_widgets_from_strategy(params)
            else:
                # Fallback: try to get parameters from an instance
                try:
                    temp_strategy = strategy_class()
                    if hasattr(temp_strategy, 'get_params'):
                        params = temp_strategy.get_params()
                        self.create_param_widgets_from_strategy(params)
                    else:
                        self.params_layout.addWidget(QLabel("No configurable parameters for this strategy"))
                except Exception as e:
                    self.logger.error(f"Error getting strategy parameters: {e}")
                    self.params_layout.addWidget(QLabel("Error loading parameters"))
                
            # Enable buttons
            self.save_preset_btn.setEnabled(True)
            self.load_preset_btn.setEnabled(True)
            
            # Update presets
            self.update_preset_combo()
            
            # Update preview
            self.update_signal_preview()
            
            self.status_update.emit(f"Loaded strategy: {strategy_name}", "success")
            
        except Exception as e:
            self.logger.error(f"Error selecting strategy: {e}")
            self.status_update.emit(f"Error loading strategy: {str(e)}", "error")
            
    def create_param_widgets(self, params):
        """Create parameter input widgets with real-time validation"""
        self.param_widgets = {}
        
        for param_name, param_config in params.items():
            # Container for this parameter
            param_frame = QFrame()
            param_frame.setStyleSheet("""
                QFrame {
                    background-color: #252525;
                    border-radius: 4px;
                    padding: 8px;
                    margin: 2px;
                }
            """)
            param_layout = QVBoxLayout()
            param_layout.setSpacing(4)
            
            # Parameter name and current value
            header_layout = QHBoxLayout()
            
            name_label = QLabel(f"<b>{param_name}</b>")
            name_label.setStyleSheet("color: #dcdcaa;")
            
            value_label = QLabel()
            value_label.setObjectName("valueLabel")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            value_label.setStyleSheet("color: #4ec9b0; font-weight: bold;")
            
            header_layout.addWidget(name_label)
            header_layout.addStretch()
            header_layout.addWidget(value_label)
            
            param_layout.addLayout(header_layout)
            
            # Slider and spinbox row
            control_layout = QHBoxLayout()
            
            min_val = param_config.get('min', 0)
            max_val = param_config.get('max', 100)
            default_val = param_config.get('default', (min_val + max_val) / 2)
            step = param_config.get('step', 1)
            
            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(int(min_val / step))
            slider.setMaximum(int(max_val / step))
            slider.setValue(int(default_val / step))
            slider.setStyleSheet("""
                QSlider::groove:horizontal {
                    background: #3e3e3e;
                    height: 6px;
                    border-radius: 3px;
                }
                QSlider::handle:horizontal {
                    background: #0e639c;
                    width: 16px;
                    height: 16px;
                    margin: -5px 0;
                    border-radius: 8px;
                }
                QSlider::handle:horizontal:hover {
                    background: #1177bb;
                }
            """)
            
            # Spinbox
            spin_box = QDoubleSpinBox()
            spin_box.setMinimum(min_val)
            spin_box.setMaximum(max_val)
            spin_box.setValue(default_val)
            spin_box.setSingleStep(step)
            spin_box.setDecimals(2 if step < 1 else 0)
            spin_box.setMaximumWidth(100)
            
            control_layout.addWidget(slider, 1)
            control_layout.addWidget(spin_box)
            
            param_layout.addLayout(control_layout)
            
            # Range label
            range_label = QLabel(f"Range: {min_val} - {max_val}")
            range_label.setStyleSheet("color: #888888; font-size: 10px;")
            param_layout.addWidget(range_label)
            
            param_frame.setLayout(param_layout)
            self.params_layout.addWidget(param_frame)
            
            # Update value label
            value_label.setText(f"{default_val:.2f}" if step < 1 else f"{int(default_val)}")
            
            # Store widgets
            self.param_widgets[param_name] = {
                'slider': slider,
                'spinbox': spin_box,
                'value_label': value_label,
                'config': param_config
            }
            
            # Connect signals for real-time updates
            slider.valueChanged.connect(
                lambda val, name=param_name: self.on_param_changed(name, val * step, 'slider')
            )
            spin_box.valueChanged.connect(
                lambda val, name=param_name: self.on_param_changed(name, val, 'spinbox')
            )
            
    def create_param_widgets_from_strategy(self, params):
        """Create parameter widgets from strategy parameters dict"""
        # Convert strategy params to the format expected by create_param_widgets
        converted_params = {}
        
        for param_name, param_info in params.items():
            if isinstance(param_info, dict):
                # Already in correct format
                converted_params[param_name] = param_info
            else:
                # Convert from simple value to dict format
                # Assume it's a default value, create reasonable min/max
                default_val = param_info
                if isinstance(default_val, int):
                    min_val = max(0, default_val // 2)
                    max_val = default_val * 2
                    step = 1
                elif isinstance(default_val, float):
                    min_val = max(0, default_val * 0.5)
                    max_val = default_val * 2.0
                    step = default_val * 0.1
                else:
                    # Skip non-numeric parameters
                    continue
                    
                converted_params[param_name] = {
                    'min': min_val,
                    'max': max_val,
                    'default': default_val,
                    'step': step
                }
        
        if converted_params:
            self.create_param_widgets(converted_params)
        else:
            self.params_layout.addWidget(QLabel("No configurable parameters for this strategy"))
            
    def clear_param_widgets(self):
        """Clear all parameter widgets"""
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.param_widgets = {}
        
    def on_param_changed(self, param_name, value, source):
        """Handle parameter value changes with real-time validation"""
        if param_name not in self.param_widgets:
            return
            
        widgets = self.param_widgets[param_name]
        step = widgets['config'].get('step', 1)
        
        # Update widgets without triggering loops
        if source == 'slider':
            widgets['spinbox'].blockSignals(True)
            widgets['spinbox'].setValue(value)
            widgets['spinbox'].blockSignals(False)
        else:  # source == 'spinbox'
            slider_val = int(value / step)
            widgets['slider'].blockSignals(True)
            widgets['slider'].setValue(slider_val)
            widgets['slider'].blockSignals(False)
            
        # Update value label
        if step < 1:
            widgets['value_label'].setText(f"{value:.2f}")
        else:
            widgets['value_label'].setText(f"{int(value)}")
            
        # Validate in real-time
        self.validate_params_realtime()
        
        # Update preview (with debounce)
        if hasattr(self, '_preview_timer'):
            self._preview_timer.stop()
        else:
            self._preview_timer = QTimer()
            self._preview_timer.setSingleShot(True)
            self._preview_timer.timeout.connect(self.update_signal_preview)
            
        self._preview_timer.start(500)  # 500ms debounce
        
    def validate_params_realtime(self):
        """Validate parameters in real-time and update UI"""
        if not self.current_strategy:
            return
            
        params = self.get_current_params()
        valid, message = self.backend.validate_params(self.current_strategy, params)
        
        if valid:
            self.validation_banner.setStyleSheet("""
                QFrame {
                    background-color: #2d2d2d;
                    border-left: 4px solid #4ec9b0;
                    border-radius: 4px;
                    padding: 8px;
                }
            """)
            self.validation_icon.setText("✓")
            self.validation_text.setText(f"Parameters valid: {message}")
            self.action_status.setText("Ready to proceed")
            self.action_status.setStyleSheet("color: #4ec9b0; font-weight: bold;")
        else:
            self.validation_banner.setStyleSheet("""
                QFrame {
                    background-color: #2d2d2d;
                    border-left: 4px solid #f48771;
                    border-radius: 4px;
                    padding: 8px;
                }
            """)
            self.validation_icon.setText("⚠️")
            self.validation_text.setText(f"Validation issue: {message}")
            self.action_status.setText("Fix validation errors")
            self.action_status.setStyleSheet("color: #f48771; font-weight: bold;")
            
    def get_current_params(self):
        """Get current parameter values"""
        params = {}
        for param_name, widgets in self.param_widgets.items():
            params[param_name] = widgets['spinbox'].value()
        return params
        
    def update_signal_preview(self):
        """Update signal preview chart and statistics"""
        if not self.current_strategy:
            return
            
        try:
            # Get current data (use research asset or production data)
            if self.current_mode == "research":
                asset = self.asset_combo.currentText().replace("/", "")
                timeframe = self.timeframe_combo.currentText()
                # TODO: Load data for selected asset and timeframe
                # For now, use available data
                data_dict = self.parent_platform.data_dict
            else:
                # Production mode: use loaded data
                data_dict = self.parent_platform.data_dict
                
            if not data_dict:
                self.chart_view.setHtml("<html><body><h3>No data available</h3></body></html>")
                return
                
            # Get number of bars for preview
            n_bars = self.preview_bars_spin.value()
            
            # Create strategy instance directly (already loaded)
            params = self.get_current_params()
            strategy = self.current_strategy(**params)
            
            # Generate signals on sample data
            df_sample = {}
            for tf, df in data_dict.items():
                if len(df) > 0:
                    df_sample[tf] = df.tail(n_bars).copy()
                    
            if not df_sample:
                return
                
            # Generate signals (in background thread to avoid UI freeze)
            self.action_status.setText("Generating preview...")
            self.action_status.setStyleSheet("color: #c586c0; font-weight: bold;")
            
            # For now, generate inline (TODO: use thread)
            signals = strategy.generate_signals(df_sample)
            
            # Create Plotly chart
            self.create_signal_chart(df_sample, signals)
            
            # Update statistics
            self.update_signal_stats(signals)
            
            self.action_status.setText("Preview updated")
            self.action_status.setStyleSheet("color: #4ec9b0; font-weight: bold;")
            
        except Exception as e:
            self.logger.error(f"Error updating signal preview: {e}")
            self.status_update.emit(f"Preview error: {str(e)}", "error")
            
    def create_signal_chart(self, data_dict, signals):
        """Create Plotly chart with candlesticks and signal markers"""
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            # Get primary timeframe data
            primary_tf = '5min' if '5min' in data_dict else list(data_dict.keys())[0]
            df = data_dict[primary_tf]
            
            # Create figure
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=('Price & Signals', 'Signal Strength')
            )
            
            # Candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price'
                ),
                row=1, col=1
            )
            
            # Add signals if available
            if hasattr(signals, 'signals') and signals.signals is not None:
                signal_df = signals.signals
                
                # Buy signals
                buy_mask = signal_df > 0  # Adjust based on your signal format
                if buy_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[buy_mask],
                            y=df.loc[buy_mask, 'Low'] * 0.995,
                            mode='markers',
                            name='Buy Signal',
                            marker=dict(symbol='triangle-up', size=12, color='#4ec9b0')
                        ),
                        row=1, col=1
                    )
                    
                # Sell signals
                sell_mask = signal_df < 0  # Adjust based on your signal format
                if sell_mask.any():
                    fig.add_trace(
                        go.Scatter(
                            x=df.index[sell_mask],
                            y=df.loc[sell_mask, 'High'] * 1.005,
                            mode='markers',
                            name='Sell Signal',
                            marker=dict(symbol='triangle-down', size=12, color='#f48771')
                        ),
                        row=1, col=1
                    )
                    
                # Signal strength subplot
                fig.add_trace(
                    go.Bar(
                        x=signal_df.index,
                        y=signal_df.values if hasattr(signal_df, 'values') else signal_df,
                        name='Signal Strength',
                        marker=dict(
                            color=signal_df.values if hasattr(signal_df, 'values') else signal_df,
                            colorscale='RdYlGn',
                            cmin=-1,
                            cmax=1
                        )
                    ),
                    row=2, col=1
                )
                
            # Update layout
            fig.update_layout(
                template='plotly_dark',
                height=500,
                showlegend=True,
                xaxis_rangeslider_visible=False,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc')
            )
            
            # Display chart
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error creating signal chart: {e}")
            
    def update_signal_stats(self, signals):
        """Update signal statistics display"""
        try:
            if not hasattr(signals, 'signals') or signals.signals is None:
                return
                
            signal_df = signals.signals
            
            # Count signals
            total = len(signal_df)
            buy_count = (signal_df > 0).sum() if hasattr(signal_df, 'sum') else 0
            sell_count = (signal_df < 0).sum() if hasattr(signal_df, 'sum') else 0
            signal_rate = ((buy_count + sell_count) / total * 100) if total > 0 else 0
            
            # Update labels
            self.total_signals_label.findChild(QLabel, "statValue").setText(str(total))
            self.buy_signals_label.findChild(QLabel, "statValue").setText(str(buy_count))
            self.sell_signals_label.findChild(QLabel, "statValue").setText(str(sell_count))
            self.signal_rate_label.findChild(QLabel, "statValue").setText(f"{signal_rate:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Error updating signal stats: {e}")
            
    def on_asset_changed(self):
        """Handle asset selection change"""
        asset = self.asset_combo.currentText()
        self.status_update.emit(f"Selected asset: {asset}", "info")
        self.update_signal_preview()
        
    def on_timeframe_changed(self):
        """Handle timeframe selection change"""
        timeframe = self.timeframe_combo.currentText()
        self.status_update.emit(f"Selected timeframe: {timeframe}", "info")
        self.update_signal_preview()
        
    def validate_strategy(self):
        """Validate current strategy configuration"""
        if not self.current_strategy:
            QMessageBox.warning(self, "Validation", "Please select a strategy first")
            return
            
        params = self.get_current_params()
        valid, message = self.backend.validate_params(self.current_strategy, params)
        
        if valid:
            QMessageBox.information(self, "Validation", f"✓ Strategy configuration is valid!\n\n{message}")
            self.status_update.emit("Validation passed", "success")
        else:
            QMessageBox.warning(self, "Validation", f"⚠️ Validation failed:\n\n{message}")
            self.status_update.emit("Validation failed", "error")
            
    def on_action_button_clicked(self):
        """Handle main action button (different for each mode)"""
        if self.current_mode == "research":
            self.save_for_backtesting()
        else:
            self.deploy_to_production()
            
    def save_for_backtesting(self):
        """Save strategy configuration for backtesting"""
        if not self.current_strategy:
            QMessageBox.warning(self, "Error", "No strategy selected")
            return
            
        # Validate first
        params = self.get_current_params()
        valid, message = self.backend.validate_params(self.current_strategy, params)
        
        if not valid:
            QMessageBox.warning(self, "Validation Error", f"Cannot save invalid configuration:\n\n{message}")
            return
            
        # Save configuration
        try:
            config = {
                'name': f"{self.current_strategy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'strategy': self.current_strategy,
                'params': params,
                'asset': self.asset_combo.currentText(),
                'timeframe': self.timeframe_combo.currentText(),
                'created': datetime.now().isoformat(),
                'mode': 'research'
            }
            
            # Save to config directory
            os.makedirs('config/strategies/research', exist_ok=True)
            filename = f"config/strategies/research/{config['name']}.json"
            
            with open(filename, 'w') as f:
                json.dump(config, f, indent=2)
                
            QMessageBox.information(
                self,
                "Saved",
                f"Strategy configuration saved!\n\nFile: {filename}\n\nYou can now run backtests with this configuration."
            )
            
            self.status_update.emit("Strategy saved for backtesting", "success")
            
        except Exception as e:
            self.logger.error(f"Error saving strategy: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save strategy:\n\n{str(e)}")
            
    def deploy_to_production(self):
        """Deploy selected strategy to live trading"""
        selected_items = self.saved_strategies_table.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "Please select a strategy to deploy")
            return
            
        row = selected_items[0].row()
        strategy_name = self.saved_strategies_table.item(row, 0).text()
        
        # Confirm deployment
        reply = QMessageBox.question(
            self,
            "Deploy to Live Trading",
            f"Are you sure you want to deploy '{strategy_name}' to live trading?\n\n"
            "This will start executing real trades on your Alpaca account.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # TODO: Implement actual deployment logic
            QMessageBox.information(
                self,
                "Deployment",
                f"Strategy '{strategy_name}' deployment initiated!\n\n"
                "Monitor the Live Trading tab for execution status."
            )
            self.status_update.emit(f"Deployed {strategy_name} to production", "success")
            
    def on_save_preset(self):
        """Save parameter preset"""
        preset_name = self.preset_name_edit.text().strip()
        if not preset_name:
            QMessageBox.warning(self, "Error", "Please enter a preset name")
            return
            
        if not self.current_strategy:
            QMessageBox.warning(self, "Error", "No strategy selected")
            return
            
        try:
            params = self.get_current_params()
            preset_data = {
                'strategy': self.current_strategy,
                'params': params,
                'created': datetime.now().isoformat()
            }
            
            os.makedirs('config/presets', exist_ok=True)
            filename = f"config/presets/{self.current_strategy}_{preset_name}.json"
            
            with open(filename, 'w') as f:
                json.dump(preset_data, f, indent=2)
                
            QMessageBox.information(self, "Success", f"Preset '{preset_name}' saved!")
            self.update_preset_combo()
            self.preset_name_edit.clear()
            
            self.status_update.emit(f"Preset saved: {preset_name}", "success")
            
        except Exception as e:
            self.logger.error(f"Error saving preset: {e}")
            QMessageBox.critical(self, "Error", f"Failed to save preset:\n\n{str(e)}")
            
    def on_load_preset(self):
        """Load parameter preset"""
        preset_file = self.preset_combo.currentText()
        if not preset_file:
            return
            
        try:
            filename = f"config/presets/{preset_file}"
            with open(filename, 'r') as f:
                preset_data = json.load(f)
                
            params = preset_data.get('params', {})
            for param_name, value in params.items():
                if param_name in self.param_widgets:
                    widgets = self.param_widgets[param_name]
                    widgets['spinbox'].setValue(value)
                    
            QMessageBox.information(self, "Success", f"Preset loaded: {preset_file}")
            self.status_update.emit(f"Preset loaded: {preset_file}", "success")
            
        except Exception as e:
            self.logger.error(f"Error loading preset: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load preset:\n\n{str(e)}")
            
    def update_preset_combo(self):
        """Update preset dropdown"""
        if not self.current_strategy:
            return
            
        try:
            import glob
            self.preset_combo.clear()
            
            pattern = f"config/presets/{self.current_strategy}_*.json"
            preset_files = glob.glob(pattern)
            
            for filepath in preset_files:
                filename = os.path.basename(filepath)
                self.preset_combo.addItem(filename)
                
        except Exception as e:
            self.logger.error(f"Error updating preset combo: {e}")
