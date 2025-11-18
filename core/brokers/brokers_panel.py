"""
UI panel for broker management and live trading.
"""

import logging
from datetime import datetime
from typing import Optional

try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                                 QTableWidget, QTableWidgetItem, QTabWidget, QCheckBox,
                                 QGroupBox, QComboBox, QSpinBox, QLineEdit,
                                 QFormLayout, QDialog, QDialogButtonBox, QMessageBox,
                                 QHeaderView)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

from .broker_manager import BrokerManager
from .broker_interfaces import BrokerConfig

if PYQT6_AVAILABLE:

    class BrokerUpdateWorker(QObject):
        """Worker for updating broker data in background."""

        finished = pyqtSignal()
        data_updated = pyqtSignal(dict)

        def __init__(self, broker_manager: BrokerManager):
            super().__init__()
            self.broker_manager = broker_manager

        def update(self):
            """Update broker data."""
            try:
                summary = self.broker_manager.get_status_summary()
                self.data_updated.emit(summary)
            finally:
                self.finished.emit()


    class BrokerConfigDialog(QDialog):
        """Dialog for configuring a broker."""

        def __init__(self, parent=None, config: Optional[BrokerConfig] = None):
            super().__init__(parent)
            self.config = config or BrokerConfig(
                name="", broker_type="alpaca", credentials={}, enabled=True
            )
            self.setWindowTitle("Configurar Broker" if config else "Nuevo Broker")
            self.setModal(True)
            self.resize(500, 400)

            self._setup_ui()
            if config:
                self._load_config()

        def _setup_ui(self):
            """Setup the dialog UI."""
            layout = QVBoxLayout(self)

            # Basic configuration group
            basic_group = QGroupBox("Configuración Básica")
            basic_layout = QFormLayout(basic_group)

            self.le_name = QLineEdit()
            basic_layout.addRow("Nombre:", self.le_name)

            self.cb_type = QComboBox()
            self.cb_type.addItems(["alpaca"])
            basic_layout.addRow("Tipo:", self.cb_type)

            self.cb_enabled = QCheckBox("Habilitado")
            self.cb_enabled.setChecked(True)
            basic_layout.addRow("", self.cb_enabled)

            layout.addWidget(basic_group)

            # Credentials group
            creds_group = QGroupBox("Credenciales")
            creds_layout = QFormLayout(creds_group)

            self.le_api_key = QLineEdit()
            self.le_api_key.setEchoMode(QLineEdit.EchoMode.Password)
            creds_layout.addRow("API Key:", self.le_api_key)

            self.le_api_secret = QLineEdit()
            self.le_api_secret.setEchoMode(QLineEdit.EchoMode.Password)
            creds_layout.addRow("API Secret:", self.le_api_secret)

            self.le_base_url = QLineEdit()
            self.le_base_url.setText("https://paper-api.alpaca.markets")
            creds_layout.addRow("Base URL:", self.le_base_url)

            layout.addWidget(creds_group)

            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def _load_config(self):
            """Load existing configuration."""
            if self.config:
                self.le_name.setText(self.config.name)
                self.cb_type.setCurrentText(self.config.broker_type)
                self.cb_enabled.setChecked(self.config.enabled)

                creds = self.config.credentials
                self.le_api_key.setText(creds.get('api_key', ''))
                self.le_api_secret.setText(creds.get('api_secret', ''))
                self.le_base_url.setText(creds.get('base_url', ''))

        def get_config(self) -> BrokerConfig:
            """Get the configured broker config."""
            return BrokerConfig(
                name=self.le_name.text(),
                broker_type=self.cb_type.currentText(),
                credentials={
                    'api_key': self.le_api_key.text(),
                    'api_secret': self.le_api_secret.text(),
                    'base_url': self.le_base_url.text()
                },
                enabled=self.cb_enabled.isChecked()
            )


    class OrderDialog(QDialog):
        """Dialog for placing orders."""

        def __init__(self, parent=None, broker_manager=None):
            super().__init__(parent)
            self.broker_manager = broker_manager
            self.setWindowTitle("Place Order")
            self.setModal(True)
            self.resize(400, 300)

            self._setup_ui()

        def _setup_ui(self):
            """Setup the order dialog UI."""
            layout = QVBoxLayout(self)

            # Order details group
            order_group = QGroupBox("Order Details")
            order_layout = QFormLayout(order_group)

            self.le_symbol = QLineEdit()
            order_layout.addRow("Symbol:", self.le_symbol)

            self.cb_side = QComboBox()
            self.cb_side.addItems(["buy", "sell"])
            order_layout.addRow("Side:", self.cb_side)

            self.cb_type = QComboBox()
            self.cb_type.addItems(["market", "limit", "stop", "stop_limit"])
            order_layout.addRow("Type:", self.cb_type)

            self.le_qty = QSpinBox()
            self.le_qty.setMinimum(1)
            self.le_qty.setMaximum(1000000)
            order_layout.addRow("Quantity:", self.le_qty)

            self.le_price = QLineEdit()
            self.le_price.setPlaceholderText("For limit/stop orders")
            order_layout.addRow("Price:", self.le_price)

            layout.addWidget(order_group)

            # Buttons
            buttons = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
            )
            buttons.accepted.connect(self.accept)
            buttons.rejected.connect(self.reject)
            layout.addWidget(buttons)

        def get_order_data(self):
            """Get the order data from the dialog."""
            return {
                'symbol': self.le_symbol.text(),
                'side': self.cb_side.currentText(),
                'type': self.cb_type.currentText(),
                'qty': self.le_qty.value(),
                'price': self.le_price.text() if self.le_price.text() else None
            }


    class BrokersPanel(QWidget):
        """Main panel for broker management and live trading."""

        def __init__(self, broker_manager: BrokerManager):
            super().__init__()
            self.broker_manager = broker_manager
            self.logger = logging.getLogger(__name__)

            self._setup_ui()
            self._connect_signals()

        def _setup_ui(self):
            """Setup the main UI."""
            layout = QVBoxLayout(self)

            # Control buttons
            controls_layout = QHBoxLayout()

            self.btn_add_broker = QPushButton("Add Broker")
            self.btn_add_broker.clicked.connect(self._add_broker)
            controls_layout.addWidget(self.btn_add_broker)

            self.btn_refresh = QPushButton("Refresh")
            self.btn_refresh.clicked.connect(self._refresh_data)
            controls_layout.addWidget(self.btn_refresh)

            controls_layout.addStretch()
            layout.addLayout(controls_layout)

            # Main tabs
            self.tabs = QTabWidget()

            # Brokers tab
            self._setup_brokers_tab()
            self.tabs.addTab(self.brokers_tab, "Brokers")

            # Orders tab
            self._setup_orders_tab()
            self.tabs.addTab(self.orders_tab, "Orders")

            # Positions tab
            self._setup_positions_tab()
            self.tabs.addTab(self.positions_tab, "Positions")

            layout.addWidget(self.tabs)

        def _setup_brokers_tab(self):
            """Setup the brokers management tab."""
            self.brokers_tab = QWidget()
            layout = QVBoxLayout(self.brokers_tab)

            self.brokers_table = QTableWidget()
            self.brokers_table.setColumnCount(4)
            self.brokers_table.setHorizontalHeaderLabels(["Name", "Type", "Status", "Actions"])
            self.brokers_table.horizontalHeader().setStretchLastSection(True)

            layout.addWidget(self.brokers_table)

        def _setup_orders_tab(self):
            """Setup the orders management tab."""
            self.orders_tab = QWidget()
            layout = QVBoxLayout(self.orders_tab)

            # Order controls
            order_controls = QHBoxLayout()

            self.btn_place_order = QPushButton("Place Order")
            self.btn_place_order.clicked.connect(self._place_order)
            order_controls.addWidget(self.btn_place_order)

            self.btn_cancel_order = QPushButton("Cancel Order")
            order_controls.addWidget(self.btn_cancel_order)

            order_controls.addStretch()
            layout.addLayout(order_controls)

            self.orders_table = QTableWidget()
            self.orders_table.setColumnCount(6)
            self.orders_table.setHorizontalHeaderLabels(["Symbol", "Side", "Type", "Qty", "Price", "Status"])
            self.orders_table.horizontalHeader().setStretchLastSection(True)

            layout.addWidget(self.orders_table)

        def _setup_positions_tab(self):
            """Setup the positions management tab."""
            self.positions_tab = QWidget()
            layout = QVBoxLayout(self.positions_tab)

            self.positions_table = QTableWidget()
            self.positions_table.setColumnCount(5)
            self.positions_table.setHorizontalHeaderLabels(["Symbol", "Qty", "Avg Price", "Current Price", "P&L"])
            self.positions_table.horizontalHeader().setStretchLastSection(True)

            layout.addWidget(self.positions_table)

        def _connect_signals(self):
            """Connect signals."""
            # Update timer
            self.update_timer = QTimer()
            self.update_timer.timeout.connect(self._refresh_data)
            self.update_timer.start(5000)  # Update every 5 seconds

        def _add_broker(self):
            """Add a new broker."""
            dialog = BrokerConfigDialog(self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                config = dialog.get_config()
                try:
                    self.broker_manager.add_broker(config)
                    self._refresh_data()
                    QMessageBox.information(self, "Success", f"Broker '{config.name}' added successfully!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to add broker: {str(e)}")

        def _place_order(self):
            """Place a new order."""
            dialog = OrderDialog(self, self.broker_manager)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                order_data = dialog.get_order_data()
                try:
                    # For now, just show the order data
                    QMessageBox.information(self, "Order", f"Order placed: {order_data}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to place order: {str(e)}")

        def _refresh_data(self):
            """Refresh all broker data."""
            try:
                # Update brokers table
                self._update_brokers_table()

                # Update orders table
                self._update_orders_table()

                # Update positions table
                self._update_positions_table()

            except Exception as e:
                self.logger.error(f"Error refreshing data: {e}")

        def _update_brokers_table(self):
            """Update the brokers table."""
            brokers = self.broker_manager.get_brokers()
            self.brokers_table.setRowCount(len(brokers))

            for row, (name, broker) in enumerate(brokers.items()):
                self.brokers_table.setItem(row, 0, QTableWidgetItem(name))
                self.brokers_table.setItem(row, 1, QTableWidgetItem(broker.__class__.__name__))
                self.brokers_table.setItem(row, 2, QTableWidgetItem("Connected" if broker.connected else "Disconnected"))

                # Actions button
                actions_widget = QWidget()
                actions_layout = QHBoxLayout(actions_widget)

                btn_connect = QPushButton("Connect" if not broker.connected else "Disconnect")
                btn_connect.clicked.connect(lambda checked, b=broker: self._toggle_broker_connection(b))
                actions_layout.addWidget(btn_connect)

                btn_remove = QPushButton("Remove")
                btn_remove.clicked.connect(lambda checked, n=name: self._remove_broker(n))
                actions_layout.addWidget(btn_remove)

                self.brokers_table.setCellWidget(row, 3, actions_widget)

        def _update_orders_table(self):
            """Update the orders table."""
            # For now, just clear the table
            self.orders_table.setRowCount(0)

        def _update_positions_table(self):
            """Update the positions table."""
            # For now, just clear the table
            self.positions_table.setRowCount(0)

        def _toggle_broker_connection(self, broker):
            """Toggle broker connection."""
            try:
                if broker.connected:
                    broker.disconnect()
                else:
                    broker.connect()
                self._refresh_data()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to toggle connection: {str(e)}")

        def _remove_broker(self, name: str):
            """Remove a broker."""
            reply = QMessageBox.question(
                self, "Remove Broker",
                f"Are you sure you want to remove broker '{name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                try:
                    self.broker_manager.remove_broker(name)
                    self._refresh_data()
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to remove broker: {str(e)}")

        def on_tab_activated(self):
            """Called when the tab is activated."""
            self._refresh_data()

else:
    # Fallback classes when PyQt6 is not available
    BrokerConfigDialog = None
    OrderDialog = None
    BrokersPanel = None