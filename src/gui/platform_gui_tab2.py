from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton,
    QDoubleSpinBox, QSlider, QGroupBox, QTextEdit, QTableWidget,
    QTableWidgetItem, QLineEdit, QHeaderView
)
from PySide6.QtCore import Qt, Signal
import logging
import pandas as pd

class Tab2StrategyConfig(QWidget):
    config_ready = Signal(dict)

    def __init__(self, parent_platform, backend):
        super().__init__()
        self.parent_platform = parent_platform
        self.backend = backend
        self.current_strategy = None
        self.param_widgets = {}
        self.logger = logging.getLogger(__name__)

        self.init_ui()
        self.populate_strategies()

    def init_ui(self):
        layout = QVBoxLayout()

        # Strategy Selection
        strategy_group = QGroupBox("Strategy Selection")
        strategy_layout = QVBoxLayout()

        self.strategy_combo = QComboBox()
        self.strategy_combo.currentTextChanged.connect(self.on_strategy_selected)

        self.strategy_desc = QTextEdit()
        self.strategy_desc.setMaximumHeight(80)
        self.strategy_desc.setReadOnly(True)
        self.strategy_desc.setPlainText("Select a strategy to view its description and parameters.")

        strategy_layout.addWidget(QLabel("Available Strategies:"))
        strategy_layout.addWidget(self.strategy_combo)
        strategy_layout.addWidget(QLabel("Description:"))
        strategy_layout.addWidget(self.strategy_desc)
        strategy_group.setLayout(strategy_layout)
        layout.addWidget(strategy_group)

        # Parameters Section
        params_group = QGroupBox("Strategy Parameters")
        self.params_layout = QVBoxLayout()
        self.params_layout.addWidget(QLabel("Select a strategy to configure parameters."))
        params_group.setLayout(self.params_layout)
        layout.addWidget(params_group)

        # Preset Management
        preset_group = QGroupBox("Preset Management")
        preset_layout = QHBoxLayout()

        self.save_preset_btn = QPushButton("Save Preset")
        self.save_preset_btn.clicked.connect(self.on_save_preset)
        self.save_preset_btn.setEnabled(False)

        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Preset name")
        self.preset_name_edit.setMaximumWidth(150)

        self.load_preset_btn = QPushButton("Load Preset")
        self.load_preset_btn.clicked.connect(self.on_load_preset)
        self.load_preset_btn.setEnabled(False)

        self.preset_combo = QComboBox()
        self.preset_combo.setMaximumWidth(150)

        preset_layout.addWidget(self.save_preset_btn)
        preset_layout.addWidget(self.preset_name_edit)
        preset_layout.addStretch()
        preset_layout.addWidget(self.load_preset_btn)
        preset_layout.addWidget(self.preset_combo)
        preset_group.setLayout(preset_layout)
        layout.addWidget(preset_group)

        # Signal Preview
        preview_group = QGroupBox("Signal Preview (First 50 signals)")
        preview_layout = QVBoxLayout()

        self.preview_table = QTableWidget()
        self.preview_table.setColumnCount(5)
        self.preview_table.setHorizontalHeaderLabels(["Timestamp", "Signal Type", "Price", "Strength", "Components"])
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.preview_table.setMaximumHeight(200)

        preview_layout.addWidget(self.preview_table)
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)

        layout.addStretch()
        self.setLayout(layout)

    def populate_strategies(self):
        try:
            strategies = self.backend.list_available_strategies()
            self.strategy_combo.clear()
            self.strategy_combo.addItems(strategies)

            if strategies:
                self.on_strategy_selected(strategies[0])

        except Exception as e:
            self.logger.error(f"Error populating strategies: {e}")
            self.strategy_combo.addItem("Error loading strategies")

    def on_strategy_selected(self, strategy_name):
        if not strategy_name or strategy_name == "Error loading strategies":
            return

        try:
            self.current_strategy = strategy_name

            # Get strategy description
            descriptions = {
                "IBS_BB": "Intraday Bias Score + Bollinger Bands strategy for crypto markets. Uses IBS to identify intraday bias and BB for mean reversion signals.",
                "MACD_ADX": "MACD + ADX momentum strategy for stocks. Combines trend strength (ADX) with momentum signals (MACD) for high-probability entries.",
                "Pairs": "Statistical arbitrage pairs trading strategy. Identifies cointegrated pairs and trades mean-reversion between them.",
                "HFT_VMA": "High-frequency Volume Moving Average strategy for forex. Uses short-term volume patterns for scalping opportunities.",
                "LSTM_ML": "LSTM neural network strategy for commodities. Machine learning model trained on price patterns for prediction-based trading."
            }

            self.strategy_desc.setPlainText(descriptions.get(strategy_name, "No description available."))

            # Clear existing parameter widgets
            self.clear_param_widgets()

            # Get parameters
            params = self.backend.get_strategy_params(strategy_name)
            if isinstance(params, dict) and 'error' not in params:
                self.create_param_widgets(params)

            # Enable preset buttons
            self.save_preset_btn.setEnabled(True)
            self.load_preset_btn.setEnabled(True)

            # Update presets
            self.update_preset_combo()

            # Generate preview signals
            self.update_signal_preview()

        except Exception as e:
            self.logger.error(f"Error selecting strategy: {e}")

    def create_param_widgets(self, params):
        self.param_widgets = {}

        for param_name, param_config in params.items():
            # Create horizontal layout for each parameter
            param_layout = QHBoxLayout()

            # Label
            label = QLabel(f"{param_name}:")
            label.setMinimumWidth(120)
            param_layout.addWidget(label)

            # Slider
            slider = QSlider(Qt.Orientation.Horizontal)
            min_val = param_config.get('min', 0)
            max_val = param_config.get('max', 100)
            default_val = param_config.get('default', (min_val + max_val) / 2)
            step = param_config.get('step', 1)

            slider.setMinimum(int(min_val / step))
            slider.setMaximum(int(max_val / step))
            slider.setValue(int(default_val / step))
            slider.setTickInterval(1)
            slider.valueChanged.connect(lambda value, name=param_name: self.on_slider_changed(name, value))

            # Spin box
            spin_box = QDoubleSpinBox()
            spin_box.setMinimum(min_val)
            spin_box.setMaximum(max_val)
            spin_box.setValue(default_val)
            spin_box.setSingleStep(step)
            spin_box.setDecimals(2 if step < 1 else 0)
            spin_box.valueChanged.connect(lambda value, name=param_name: self.on_spinbox_changed(name, value))

            # Value label
            value_label = QLabel(f"{default_val:.2f}")
            value_label.setMinimumWidth(60)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

            param_layout.addWidget(slider)
            param_layout.addWidget(spin_box)
            param_layout.addWidget(value_label)

            # Store widgets
            self.param_widgets[param_name] = {
                'slider': slider,
                'spinbox': spin_box,
                'label': value_label,
                'config': param_config
            }

            # Add to layout
            self.params_layout.addLayout(param_layout)

    def clear_param_widgets(self):
        # Remove all parameter widgets from layout
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            if item.layout():
                # Clear sub-layout
                sub_layout = item.layout()
                while sub_layout.count():
                    sub_item = sub_layout.takeAt(0)
                    if sub_item.widget():
                        sub_item.widget().deleteLater()
                item.layout().deleteLater()
            elif item.widget():
                item.widget().deleteLater()

        self.param_widgets = {}

    def on_slider_changed(self, param_name, slider_value):
        if param_name not in self.param_widgets:
            return

        widgets = self.param_widgets[param_name]
        step = widgets['config'].get('step', 1)
        actual_value = slider_value * step

        # Update spinbox without triggering its signal
        widgets['spinbox'].blockSignals(True)
        widgets['spinbox'].setValue(actual_value)
        widgets['spinbox'].blockSignals(False)

        # Update label
        widgets['label'].setText(f"{actual_value:.2f}")

        # Update signal preview
        self.update_signal_preview()

    def on_spinbox_changed(self, param_name, value):
        if param_name not in self.param_widgets:
            return

        widgets = self.param_widgets[param_name]
        step = widgets['config'].get('step', 1)

        # Update slider
        slider_value = int(value / step)
        widgets['slider'].blockSignals(True)
        widgets['slider'].setValue(slider_value)
        widgets['slider'].blockSignals(False)

        # Update label
        widgets['label'].setText(f"{value:.2f}")

        # Update signal preview
        self.update_signal_preview()

    def get_current_params(self):
        params = {}
        for param_name, widgets in self.param_widgets.items():
            params[param_name] = widgets['spinbox'].value()
        return params

    def validate_params(self):
        if not self.current_strategy:
            return False, "No strategy selected"

        params = self.get_current_params()
        valid, msg = self.backend.validate_params(self.current_strategy, params)

        if valid:
            # Emit config ready signal
            config = {
                'strategy_name': self.current_strategy,
                'strategy_class': self.backend.load_strategy_module(self.current_strategy),
                'params': params
            }
            self.config_ready.emit(config)

        return valid, msg

    def update_signal_preview(self):
        try:
            if not self.current_strategy or not self.parent_platform.data_dict:
                self.preview_table.setRowCount(0)
                return

            # Get current parameters
            params = self.get_current_params()

            # Load strategy class
            strategy_module = self.backend.load_strategy_module(self.current_strategy)
            if isinstance(strategy_module, dict) and 'error' in strategy_module:
                return

            # Create strategy instance
            try:
                strategy = strategy_module(**params)
            except Exception as e:
                self.logger.error(f"Error creating strategy instance: {e}")
                return

            # Generate signals on first 1000 bars
            df_5m = self.parent_platform.data_dict.get('5min')
            if df_5m is None or len(df_5m) < 100:
                return

            df_sample = df_5m.head(1000)
            signals = strategy.generate_signals({'5min': df_sample})

            # Update table
            self.preview_table.setRowCount(0)

            if hasattr(signals, 'signals') and signals.signals is not None:
                signal_data = signals.signals.head(50)
                for i, (_, signal) in enumerate(signal_data.iterrows()):
                    self.preview_table.insertRow(i)

                    # Timestamp
                    timestamp_item = QTableWidgetItem(str(signal.name))
                    self.preview_table.setItem(i, 0, timestamp_item)

                    # Signal Type (placeholder)
                    signal_type = "BUY" if i % 2 == 0 else "SELL"
                    type_item = QTableWidgetItem(signal_type)
                    type_item.setBackground(Qt.GlobalColor.green if signal_type == "BUY" else Qt.GlobalColor.red)
                    self.preview_table.setItem(i, 1, type_item)

                    # Price
                    price_item = QTableWidgetItem(f"{df_sample.loc[signal.name, 'Close']:.2f}")
                    self.preview_table.setItem(i, 2, price_item)

                    # Strength (placeholder)
                    strength_item = QTableWidgetItem("4.5")
                    self.preview_table.setItem(i, 3, strength_item)

                    # Components (placeholder)
                    components_item = QTableWidgetItem("IBS: 0.7, BB: -1.2")
                    self.preview_table.setItem(i, 4, components_item)

        except Exception as e:
            self.logger.error(f"Error updating signal preview: {e}")

    def on_save_preset(self):
        preset_name = self.preset_name_edit.text().strip()
        if not preset_name:
            self.show_message("Error", "Please enter a preset name")
            return

        try:
            params = self.get_current_params()
            preset_data = {
                'strategy': self.current_strategy,
                'params': params,
                'created': str(pd.Timestamp.now())
            }

            # Save to file
            import json
            import os
            os.makedirs('config/presets', exist_ok=True)

            filename = f"config/presets/{self.current_strategy}_{preset_name}.json"
            with open(filename, 'w') as f:
                json.dump(preset_data, f, indent=2)

            self.show_message("Success", f"Preset '{preset_name}' saved successfully")
            self.update_preset_combo()

        except Exception as e:
            self.logger.error(f"Error saving preset: {e}")
            self.show_message("Error", f"Failed to save preset: {str(e)}")

    def on_load_preset(self):
        preset_file = self.preset_combo.currentText()
        if not preset_file:
            return

        try:
            import json
            filename = f"config/presets/{preset_file}"

            with open(filename, 'r') as f:
                preset_data = json.load(f)

            # Load parameters
            params = preset_data.get('params', {})
            for param_name, value in params.items():
                if param_name in self.param_widgets:
                    widgets = self.param_widgets[param_name]
                    widgets['spinbox'].setValue(value)
                    # This will trigger the slider update via signal

            self.show_message("Success", f"Preset '{preset_file}' loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading preset: {e}")
            self.show_message("Error", f"Failed to load preset: {str(e)}")

    def update_preset_combo(self):
        if not self.current_strategy:
            return

        try:
            import os
            import glob

            self.preset_combo.clear()
            pattern = f"config/presets/{self.current_strategy}_*.json"
            preset_files = glob.glob(pattern)

            for file_path in preset_files:
                filename = os.path.basename(file_path)
                self.preset_combo.addItem(filename)

        except Exception as e:
            self.logger.error(f"Error updating preset combo: {e}")

    def show_message(self, title, message):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, title, message)