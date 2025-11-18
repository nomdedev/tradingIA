"""
UI panel for broker management and live trading.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                                 QTableWidget, QTableWidgetItem, QTabWidget, QCheckBox,
                                 QGroupBox, QComboBox, QSpinBox, QTextEdit, QSplitter,
                                 QMessageBox, QHeaderView, QProgressBar, QLineEdit,
                                 QFormLayout, QDialog, QDialogButtonBox)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
    from PyQt6.QtGui import QColor, QFont, QIcon
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False

from .broker_manager import BrokerManager
from .broker_interfaces import BrokerConfig, Order, Position, Account, OrderSide, OrderType

if PYQT6_AVAILABLE:

    class BrokerUpdateWorker(QObject):
        """Worker for updating broker data in background."""

        finished = pyqtSignal()
        data_updated = pyqtSignal(dict)  # Broker status summary

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
        """Setup configuration UI."""
        layout = QVBoxLayout(self)

        # Basic settings
        basic_group = QGroupBox("Configuración Básica")
        basic_layout = QFormLayout(basic_group)

        self.le_name = QLineEdit()
        self.le_name.setPlaceholderText("Nombre del broker")
        basic_layout.addRow("Nombre:", self.le_name)

        self.cb_type = QComboBox()
        self.cb_type.addItems(["alpaca"])
        basic_layout.addRow("Tipo:", self.cb_type)

        self.cb_enabled = QCheckBox("Habilitado")
        self.cb_enabled.setChecked(True)
        basic_layout.addRow("Estado:", self.cb_enabled)

        layout.addWidget(basic_group)

        # Credentials
        creds_group = QGroupBox("Credenciales")
        creds_layout = QFormLayout(creds_group)

        self.le_api_key = QLineEdit()
        self.le_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        creds_layout.addRow("API Key:", self.le_api_key)

        self.le_api_secret = QLineEdit()
        self.le_api_secret.setEchoMode(QLineEdit.EchoMode.Password)
        creds_layout.addRow("API Secret:", self.le_api_secret)

        self.le_base_url = QLineEdit()
        self.le_base_url.setText("https://api.alpaca.markets")
        self.le_base_url.setPlaceholderText("URL base de la API")
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
        self.le_name.setText(self.config.name)
        self.cb_type.setCurrentText(self.config.broker_type)
        self.cb_enabled.setChecked(self.config.enabled)

        creds = self.config.credentials
        self.le_api_key.setText(creds.get('api_key', ''))
        self.le_api_secret.setText(creds.get('api_secret', ''))
        self.le_base_url.setText(creds.get('base_url', 'https://api.alpaca.markets'))

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

    def __init__(self, broker_manager: BrokerManager, parent=None):
        super().__init__(parent)
        self.broker_manager = broker_manager
        self.setWindowTitle("Nueva Orden")
        self.setModal(True)
        self.resize(400, 300)

        self._setup_ui()

    def _setup_ui(self):
        """Setup order UI."""
        layout = QVBoxLayout(self)

        form_layout = QFormLayout()

        # Broker selection
        self.cb_broker = QComboBox()
        self.cb_broker.addItems(self.broker_manager.get_available_brokers())
        form_layout.addRow("Broker:", self.cb_broker)

        # Symbol
        self.le_symbol = QLineEdit()
        self.le_symbol.setPlaceholderText("Ej: AAPL, BTC/USD")
        form_layout.addRow("Símbolo:", self.le_symbol)

        # Side
        self.cb_side = QComboBox()
        self.cb_side.addItems(["buy", "sell"])
        form_layout.addRow("Lado:", self.cb_side)

        # Quantity
        self.sb_quantity = QSpinBox()
        self.sb_quantity.setRange(1, 1000000)
        form_layout.addRow("Cantidad:", self.sb_quantity)

        # Order type
        self.cb_type = QComboBox()
        self.cb_type.addItems(["market", "limit", "stop", "stop_limit"])
        self.cb_type.currentTextChanged.connect(self._on_order_type_changed)
        form_layout.addRow("Tipo:", self.cb_type)

        # Price fields (shown/hidden based on type)
        self.le_limit_price = QLineEdit()
        self.le_limit_price.setPlaceholderText("Precio límite")
        form_layout.addRow("Precio Límite:", self.le_limit_price)

        self.le_stop_price = QLineEdit()
        self.le_stop_price.setPlaceholderText("Precio stop")
        form_layout.addRow("Precio Stop:", self.le_stop_price)

        layout.addLayout(form_layout)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._place_order)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._on_order_type_changed("market")

    def _on_order_type_changed(self, order_type: str):
        """Handle order type change."""
        self.le_limit_price.setVisible(order_type in ["limit", "stop_limit"])
        self.le_stop_price.setVisible(order_type in ["stop", "stop_limit"])

    def _place_order(self):
        """Place the order."""
        try:
            broker_name = self.cb_broker.currentText()
            symbol = self.le_symbol.text().upper()
            side = self.cb_side.currentText()
            quantity = self.sb_quantity.value()
            order_type = self.cb_type.currentText()

            kwargs = {}
            if order_type in ["limit", "stop_limit"] and self.le_limit_price.text():
                kwargs['limit_price'] = float(self.le_limit_price.text())
            if order_type in ["stop", "stop_limit"] and self.le_stop_price.text():
                kwargs['stop_price'] = float(self.le_stop_price.text())

            order = self.broker_manager.place_order(
                broker_name, symbol, side, quantity, order_type, **kwargs
            )

            if order:
                QMessageBox.information(self, "Éxito", f"Orden colocada: {order.id}")
                self.accept()
            else:
                QMessageBox.critical(self, "Error", "No se pudo colocar la orden")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al colocar orden: {str(e)}")


class BrokersPanel(QWidget):
    """Main panel for broker management."""

    def __init__(self, broker_manager: BrokerManager):
        super().__init__()
        self.broker_manager = broker_manager
        self.logger = logging.getLogger(__name__)

        # Update worker
        self.update_worker = BrokerUpdateWorker(broker_manager)
        self.update_worker.data_updated.connect(self._on_data_updated)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._trigger_update)
        self.update_timer.start(10000)  # Update every 10 seconds

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Gestión de Brokers y Trading en Vivo")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        # Control buttons
        self.btn_add_broker = QPushButton("Agregar Broker")
        self.btn_refresh = QPushButton("Actualizar")
        self.btn_place_order = QPushButton("Nueva Orden")

        header_layout.addWidget(self.btn_add_broker)
        header_layout.addWidget(self.btn_refresh)
        header_layout.addWidget(self.btn_place_order)

        layout.addLayout(header_layout)

        # Main content
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Status overview
        status_group = QGroupBox("Estado de Brokers")
        status_layout = QVBoxLayout(status_group)

        self.status_table = QTableWidget()
        self._setup_status_table()
        status_layout.addWidget(self.status_table)

        splitter.addWidget(status_group)

        # Positions and orders tabs
        tabs = QTabWidget()

        # Positions tab
        positions_widget = QWidget()
        positions_layout = QVBoxLayout(positions_widget)

        self.positions_table = QTableWidget()
        self._setup_positions_table()
        positions_layout.addWidget(self.positions_table)

        tabs.addTab(positions_widget, "Posiciones")

        # Orders tab
        orders_widget = QWidget()
        orders_layout = QVBoxLayout(orders_widget)

        self.orders_table = QTableWidget()
        self._setup_orders_table()
        orders_layout.addWidget(self.orders_table)

        # Order buttons
        order_btn_layout = QHBoxLayout()
        self.btn_cancel_order = QPushButton("Cancelar Orden")
        order_btn_layout.addWidget(self.btn_cancel_order)
        order_btn_layout.addStretch()
        orders_layout.addLayout(order_btn_layout)

        tabs.addTab(orders_widget, "Órdenes")

        splitter.addWidget(tabs)

        # Set splitter proportions
        splitter.setSizes([200, 400])

        layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("Listo")
        layout.addWidget(self.status_label)

    def _setup_status_table(self):
        """Setup the broker status table."""
        self.status_table.setColumnCount(5)
        self.status_table.setHorizontalHeaderLabels([
            "Broker", "Estado", "Posiciones", "Órdenes", "Tipo"
        ])

        header = self.status_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self.status_table.setColumnWidth(0, 100)
        self.status_table.setColumnWidth(1, 80)
        self.status_table.setColumnWidth(2, 80)
        self.status_table.setColumnWidth(3, 80)

        self.status_table.setAlternatingRowColors(True)

    def _setup_positions_table(self):
        """Setup the positions table."""
        self.positions_table.setColumnCount(7)
        self.positions_table.setHorizontalHeaderLabels([
            "Broker", "Símbolo", "Lado", "Cantidad", "Precio Promedio",
            "Precio Actual", "PnL No Realizado"
        ])

        header = self.positions_table.horizontalHeader()
        for i in range(7):
            if i == 0:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                self.positions_table.setColumnWidth(i, 80)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        self.positions_table.setAlternatingRowColors(True)

    def _setup_orders_table(self):
        """Setup the orders table."""
        self.orders_table.setColumnCount(8)
        self.orders_table.setHorizontalHeaderLabels([
            "Broker", "ID", "Símbolo", "Lado", "Tipo", "Cantidad",
            "Precio", "Estado"
        ])

        header = self.orders_table.horizontalHeader()
        for i in range(8):
            if i in [0, 1, 3, 4, 7]:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Fixed)
                widths = [80, 120, 0, 60, 80, 0, 0, 80]
                self.orders_table.setColumnWidth(i, widths[i])
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)

        self.orders_table.setAlternatingRowColors(True)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_add_broker.clicked.connect(self._add_broker)
        self.btn_refresh.clicked.connect(self._trigger_update)
        self.btn_place_order.clicked.connect(self._place_order)
        self.btn_cancel_order.clicked.connect(self._cancel_selected_order)

    def _trigger_update(self):
        """Trigger background update of broker data."""
        if hasattr(self.update_worker, 'update'):
            self.update_worker.update()

    def _on_data_updated(self, summary: dict):
        """Handle broker data update."""
        self._update_status_table(summary)
        self._update_positions_table()
        self._update_orders_table()

        # Update status
        connected = summary.get('connected_brokers', 0)
        total = summary.get('total_brokers', 0)
        self.status_label.setText(f"{connected}/{total} brokers conectados")

    def _update_status_table(self, summary: dict):
        """Update the broker status table."""
        brokers = summary.get('brokers', {})
        self.status_table.setRowCount(len(brokers))

        for row, (name, info) in enumerate(brokers.items()):
            # Broker name
            name_item = QTableWidgetItem(name)
            self.status_table.setItem(row, 0, name_item)

            # Status
            status = "Conectado" if info.get('connected') else "Desconectado"
            status_item = QTableWidgetItem(status)
            if info.get('connected'):
                status_item.setBackground(QColor(200, 255, 200))  # Light green
            else:
                status_item.setBackground(QColor(255, 200, 200))  # Light red
            self.status_table.setItem(row, 1, status_item)

            # Positions
            pos_count = info.get('positions', 0)
            pos_item = QTableWidgetItem(str(pos_count))
            self.status_table.setItem(row, 2, pos_item)

            # Orders
            order_count = info.get('orders', 0)
            order_item = QTableWidgetItem(str(order_count))
            self.status_table.setItem(row, 3, order_item)

            # Type
            type_item = QTableWidgetItem(info.get('type', 'Unknown'))
            self.status_table.setItem(row, 4, type_item)

    def _update_positions_table(self):
        """Update the positions table."""
        all_positions = self.broker_manager.get_all_positions()
        total_positions = sum(len(positions) for positions in all_positions.values())

        self.positions_table.setRowCount(total_positions)

        row = 0
        for broker_name, positions in all_positions.items():
            for position in positions:
                # Broker
                broker_item = QTableWidgetItem(broker_name)
                self.positions_table.setItem(row, 0, broker_item)

                # Symbol
                symbol_item = QTableWidgetItem(position.symbol)
                self.positions_table.setItem(row, 1, symbol_item)

                # Side
                side_item = QTableWidgetItem(position.side.value.title())
                self.positions_table.setItem(row, 2, side_item)

                # Quantity
                qty_item = QTableWidgetItem(f"{position.quantity:.4f}")
                self.positions_table.setItem(row, 3, qty_item)

                # Avg Price
                avg_item = QTableWidgetItem(f"{position.avg_price:.4f}")
                self.positions_table.setItem(row, 4, avg_item)

                # Current Price
                current_item = QTableWidgetItem(f"{position.current_price:.4f}")
                self.positions_table.setItem(row, 5, current_item)

                # PnL
                pnl_item = QTableWidgetItem(f"{position.unrealized_pnl:.2f}")
                if position.unrealized_pnl >= 0:
                    pnl_item.setBackground(QColor(200, 255, 200))  # Green
                else:
                    pnl_item.setBackground(QColor(255, 200, 200))  # Red
                self.positions_table.setItem(row, 6, pnl_item)

                row += 1

    def _update_orders_table(self):
        """Update the orders table."""
        all_orders = self.broker_manager.get_all_orders()
        total_orders = sum(len(orders) for orders in all_orders.values())

        self.orders_table.setRowCount(total_orders)

        row = 0
        for broker_name, orders in all_orders.items():
            for order in orders:
                # Broker
                broker_item = QTableWidgetItem(broker_name)
                self.orders_table.setItem(row, 0, broker_item)

                # Order ID
                id_item = QTableWidgetItem(order.id[:12] + "..." if len(order.id) > 12 else order.id)
                self.orders_table.setItem(row, 1, id_item)

                # Symbol
                symbol_item = QTableWidgetItem(order.symbol)
                self.orders_table.setItem(row, 2, symbol_item)

                # Side
                side_item = QTableWidgetItem(order.side.value.upper())
                self.orders_table.setItem(row, 3, side_item)

                # Type
                type_item = QTableWidgetItem(order.type.value.replace('_', ' ').title())
                self.orders_table.setItem(row, 4, type_item)

                # Quantity
                qty_item = QTableWidgetItem(f"{order.quantity:.4f}")
                self.orders_table.setItem(row, 5, qty_item)

                # Price
                price_text = ""
                if order.price:
                    price_text = f"{order.price:.4f}"
                elif order.limit_price:
                    price_text = f"Lim {order.limit_price:.4f}"
                elif order.stop_price:
                    price_text = f"Stop {order.stop_price:.4f}"
                price_item = QTableWidgetItem(price_text)
                self.orders_table.setItem(row, 6, price_item)

                # Status
                status_item = QTableWidgetItem(order.status.value.replace('_', ' ').title())
                self._set_status_color(status_item, order.status)
                self.orders_table.setItem(row, 7, status_item)

                row += 1

    def _set_status_color(self, item: QTableWidgetItem, status):
        """Set color based on order status."""
        from .broker_interfaces import OrderStatus
        if status == OrderStatus.FILLED:
            item.setBackground(QColor(200, 255, 200))  # Green
        elif status == OrderStatus.PARTIALLY_FILLED:
            item.setBackground(QColor(255, 255, 200))  # Yellow
        elif status in [OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            item.setBackground(QColor(255, 200, 200))  # Red
        else:
            item.setBackground(QColor(200, 200, 255))  # Blue

    def _add_broker(self):
        """Add a new broker."""
        dialog = BrokerConfigDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            config = dialog.get_config()
            if self.broker_manager.add_broker(config):
                QMessageBox.information(self, "Éxito", f"Broker '{config.name}' agregado correctamente")
                self._trigger_update()
            else:
                QMessageBox.critical(self, "Error", "No se pudo agregar el broker")

    def _place_order(self):
        """Place a new order."""
        if not self.broker_manager.get_available_brokers():
            QMessageBox.warning(self, "Sin Brokers", "No hay brokers configurados")
            return

        dialog = OrderDialog(self.broker_manager, self)
        dialog.exec()
        self._trigger_update()

    def _cancel_selected_order(self):
        """Cancel the selected order."""
        selected_items = self.orders_table.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Sin selección", "Seleccione una orden para cancelar")
            return

        # Get order ID from selected row
        row = selected_items[0].row()
        broker_item = self.orders_table.item(row, 0)
        id_item = self.orders_table.item(row, 1)

        if broker_item and id_item:
            broker_name = broker_item.text()
            order_id = id_item.text()

            # Note: This is a simplified version. In reality, we'd need the full order ID
            # For now, just show a message
            QMessageBox.information(self, "Función no implementada",
                                   "La cancelación de órdenes requiere el ID completo de la orden")

    def closeEvent(self, event):
        """Handle widget close event."""
        self.update_timer.stop()
        super().closeEvent(event)


# Import here to avoid circular imports
if PYQT6_AVAILABLE:
    pass  # Already imported above