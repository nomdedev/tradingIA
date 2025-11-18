"""
UI panel for managing alerts and notifications.
"""

import logging
from datetime import datetime
from typing import Optional

try:
    from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
                                 QTableWidget, QTableWidgetItem, QTabWidget, QCheckBox,
                                 QGroupBox, QComboBox, QSpinBox, QTextEdit, QSplitter,
                                 QMessageBox, QHeaderView, QProgressBar, QLineEdit)
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QObject
    from PyQt6.QtGui import QColor, QFont
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    # Don't define the classes if PyQt6 is not available
    raise ImportError("PyQt6 is required for the alerts panel")

from .alert_manager import AlertManager
from .alert_types import Alert, AlertRule, AlertType, AlertSeverity, NotificationMethod, AlertConfig


class AlertUpdateWorker(QObject):
    """Worker for updating alerts in background."""

    finished = pyqtSignal()
    alerts_updated = pyqtSignal(list)  # List of active alerts

    def __init__(self, alert_manager: AlertManager):
        super().__init__()
        self.alert_manager = alert_manager

    def update(self):
        """Update alerts data."""
        try:
            active_alerts = self.alert_manager.get_active_alerts()
            self.alerts_updated.emit(active_alerts)
        finally:
            self.finished.emit()


class AlertsPanel(QWidget):
    """Main panel for alerts management."""

    alert_selected = pyqtSignal(Alert)

    def __init__(self, alert_manager: AlertManager):
        super().__init__()
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)

        # Update worker
        self.update_worker = AlertUpdateWorker(alert_manager)
        self.update_worker.alerts_updated.connect(self._on_alerts_updated)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._trigger_update)
        self.update_timer.start(5000)  # Update every 5 seconds

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)

        # Header
        header_layout = QHBoxLayout()
        title = QLabel("Sistema de Alertas y Notificaciones")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        header_layout.addWidget(title)
        header_layout.addStretch()

        # Control buttons
        self.btn_refresh = QPushButton("Actualizar")
        self.btn_clear_history = QPushButton("Limpiar Historial")
        self.btn_configure = QPushButton("Configurar")

        header_layout.addWidget(self.btn_refresh)
        header_layout.addWidget(self.btn_clear_history)
        header_layout.addWidget(self.btn_configure)

        layout.addLayout(header_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Active alerts section
        active_group = QGroupBox("Alertas Activas")
        active_layout = QVBoxLayout(active_group)

        self.active_alerts_table = QTableWidget()
        self._setup_alerts_table(self.active_alerts_table)
        active_layout.addWidget(self.active_alerts_table)

        # Active alerts buttons
        active_btn_layout = QHBoxLayout()
        self.btn_acknowledge = QPushButton("Marcar como Leído")
        self.btn_acknowledge_all = QPushButton("Marcar Todas como Leídas")
        active_btn_layout.addWidget(self.btn_acknowledge)
        active_btn_layout.addWidget(self.btn_acknowledge_all)
        active_btn_layout.addStretch()
        active_layout.addLayout(active_btn_layout)

        splitter.addWidget(active_group)

        # History section
        history_group = QGroupBox("Historial de Alertas")
        history_layout = QVBoxLayout(history_group)

        self.history_table = QTableWidget()
        self._setup_alerts_table(self.history_table)
        history_layout.addWidget(self.history_table)

        splitter.addWidget(history_group)

        # Set splitter proportions
        splitter.setSizes([300, 300])

        layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("Listo")
        layout.addWidget(self.status_label)

    def _setup_alerts_table(self, table: QTableWidget):
        """Setup an alerts table."""
        table.setColumnCount(6)
        table.setHorizontalHeaderLabels([
            "Severidad", "Tipo", "Título", "Mensaje", "Fuente", "Tiempo"
        ])

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)

        table.setColumnWidth(0, 80)
        table.setColumnWidth(1, 100)
        table.setColumnWidth(4, 100)
        table.setColumnWidth(5, 120)

        table.setAlternatingRowColors(True)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

    def _connect_signals(self):
        """Connect UI signals."""
        self.btn_refresh.clicked.connect(self._trigger_update)
        self.btn_clear_history.clicked.connect(self._clear_history)
        self.btn_configure.clicked.connect(self._show_configuration)
        self.btn_acknowledge.clicked.connect(self._acknowledge_selected)
        self.btn_acknowledge_all.clicked.connect(self._acknowledge_all)

        self.active_alerts_table.itemSelectionChanged.connect(self._on_alert_selection_changed)
        self.history_table.itemSelectionChanged.connect(self._on_alert_selection_changed)

    def _trigger_update(self):
        """Trigger background update of alerts."""
        if hasattr(self.update_worker, 'update'):
            self.update_worker.update()

    def _on_alerts_updated(self, active_alerts: list[Alert]):
        """Handle alerts update."""
        self._update_active_alerts_table(active_alerts)
        self._update_history_table()

        # Update status
        active_count = len(active_alerts)
        self.status_label.setText(f"{active_count} alertas activas")

    def _update_active_alerts_table(self, alerts: list[Alert]):
        """Update the active alerts table."""
        self.active_alerts_table.setRowCount(len(alerts))

        for row, alert in enumerate(alerts):
            # Severity
            severity_item = QTableWidgetItem(alert.severity.value.upper())
            severity_item.setData(Qt.ItemDataRole.UserRole, alert)
            self._set_severity_color(severity_item, alert.severity)
            self.active_alerts_table.setItem(row, 0, severity_item)

            # Type
            type_item = QTableWidgetItem(alert.type.value.replace('_', ' ').title())
            self.active_alerts_table.setItem(row, 1, type_item)

            # Title
            title_item = QTableWidgetItem(alert.title)
            self.active_alerts_table.setItem(row, 2, title_item)

            # Message (truncated)
            message = alert.message
            if len(message) > 50:
                message = message[:47] + "..."
            message_item = QTableWidgetItem(message)
            self.active_alerts_table.setItem(row, 3, message_item)

            # Source
            source_item = QTableWidgetItem(alert.source)
            self.active_alerts_table.setItem(row, 4, source_item)

            # Time
            time_str = alert.timestamp.strftime("%H:%M:%S")
            time_item = QTableWidgetItem(time_str)
            self.active_alerts_table.setItem(row, 5, time_item)

    def _update_history_table(self):
        """Update the history table."""
        history = self.alert_manager.get_alert_history(limit=100)
        self.history_table.setRowCount(len(history))

        for row, alert in enumerate(reversed(history)):  # Most recent first
            # Severity
            severity_item = QTableWidgetItem(alert.severity.value.upper())
            severity_item.setData(Qt.ItemDataRole.UserRole, alert)
            self._set_severity_color(severity_item, alert.severity)
            self.history_table.setItem(row, 0, severity_item)

            # Type
            type_item = QTableWidgetItem(alert.type.value.replace('_', ' ').title())
            self.history_table.setItem(row, 1, type_item)

            # Title
            title_item = QTableWidgetItem(alert.title)
            self.history_table.setItem(row, 2, title_item)

            # Message (truncated)
            message = alert.message
            if len(message) > 50:
                message = message[:47] + "..."
            message_item = QTableWidgetItem(message)
            self.history_table.setItem(row, 3, message_item)

            # Source
            source_item = QTableWidgetItem(alert.source)
            self.history_table.setItem(row, 4, source_item)

            # Time
            time_str = alert.timestamp.strftime("%d/%m %H:%M")
            if alert.acknowledged:
                time_str += " ✓"
            time_item = QTableWidgetItem(time_str)
            self.history_table.setItem(row, 5, time_item)

    def _set_severity_color(self, item: QTableWidgetItem, severity: AlertSeverity):
        """Set color based on alert severity."""
        if severity == AlertSeverity.CRITICAL:
            item.setBackground(QColor(255, 100, 100))  # Red
        elif severity == AlertSeverity.HIGH:
            item.setBackground(QColor(255, 200, 100))  # Orange
        elif severity == AlertSeverity.MEDIUM:
            item.setBackground(QColor(255, 255, 100))  # Yellow
        else:  # LOW
            item.setBackground(QColor(200, 255, 200))  # Light green

    def _on_alert_selection_changed(self):
        """Handle alert selection change."""
        # Get selected alert from either table
        current_table = self.sender()
        if not current_table:
            return

        selected_items = current_table.selectedItems()
        if not selected_items:
            return

        # Get the alert from the first column
        first_item = selected_items[0]
        alert = first_item.data(Qt.ItemDataRole.UserRole)
        if alert:
            self.alert_selected.emit(alert)

    def _acknowledge_selected(self):
        """Acknowledge selected alerts."""
        selected_rows = set()
        for item in self.active_alerts_table.selectedItems():
            selected_rows.add(item.row())

        if not selected_rows:
            QMessageBox.information(self, "Sin selección", "Seleccione alertas para marcar como leídas.")
            return

        for row in selected_rows:
            severity_item = self.active_alerts_table.item(row, 0)
            if severity_item:
                alert = severity_item.data(Qt.ItemDataRole.UserRole)
                if alert:
                    self.alert_manager.acknowledge_alert(alert.id)

        self._trigger_update()

    def _acknowledge_all(self):
        """Acknowledge all active alerts."""
        active_alerts = self.alert_manager.get_active_alerts()
        for alert in active_alerts:
            self.alert_manager.acknowledge_alert(alert.id)

        self._trigger_update()

    def _clear_history(self):
        """Clear alert history."""
        reply = QMessageBox.question(
            self, "Confirmar",
            "¿Está seguro de que desea limpiar el historial de alertas?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.alert_manager.clear_alert_history()
            self._trigger_update()

    def _show_configuration(self):
        """Show alerts configuration dialog."""
        config_dialog = AlertConfigDialog(self.alert_manager, self)
        config_dialog.exec()

    def closeEvent(self, event):
        """Handle widget close event."""
        self.update_timer.stop()
        super().closeEvent(event)


class AlertConfigDialog(QWidget):
    """Dialog for configuring alerts."""

    def __init__(self, alert_manager: AlertManager, parent=None):
        super().__init__(parent)
        self.alert_manager = alert_manager
        self.setWindowTitle("Configuración de Alertas")
        self.setModal(True)
        self.resize(500, 600)

        self._setup_ui()
        self._load_config()

    def _setup_ui(self):
        """Setup configuration UI."""
        layout = QVBoxLayout(self)

        # General settings
        general_group = QGroupBox("Configuración General")
        general_layout = QVBoxLayout(general_group)

        self.cb_enabled = QCheckBox("Sistema de alertas habilitado")
        self.cb_sound = QCheckBox("Notificaciones de sonido")
        self.cb_gui = QCheckBox("Notificaciones en GUI")
        self.cb_log = QCheckBox("Registrar alertas")

        general_layout.addWidget(self.cb_enabled)
        general_layout.addWidget(self.cb_sound)
        general_layout.addWidget(self.cb_gui)
        general_layout.addWidget(self.cb_log)

        # Auto-acknowledge
        auto_layout = QHBoxLayout()
        auto_layout.addWidget(QLabel("Auto-marcar como leído después de"))
        self.sb_auto_ack = QSpinBox()
        self.sb_auto_ack.setRange(0, 1440)  # 0 = never, max 24 hours
        self.sb_auto_ack.setSuffix(" minutos")
        auto_layout.addWidget(self.sb_auto_ack)
        auto_layout.addStretch()
        general_layout.addLayout(auto_layout)

        layout.addWidget(general_group)

        # Email settings
        email_group = QGroupBox("Configuración de Email")
        email_layout = QVBoxLayout(email_group)

        self.cb_email_enabled = QCheckBox("Notificaciones por email")

        email_form_layout = QVBoxLayout()

        # Server settings
        server_layout = QHBoxLayout()
        server_layout.addWidget(QLabel("Servidor SMTP:"))
        self.le_smtp_server = QLineEdit()
        self.le_smtp_server.setPlaceholderText("smtp.gmail.com")
        server_layout.addWidget(self.le_smtp_server)
        server_layout.addWidget(QLabel("Puerto:"))
        self.sb_smtp_port = QSpinBox()
        self.sb_smtp_port.setRange(1, 65535)
        self.sb_smtp_port.setValue(587)
        server_layout.addWidget(self.sb_smtp_port)
        email_form_layout.addLayout(server_layout)

        # Credentials
        user_layout = QHBoxLayout()
        user_layout.addWidget(QLabel("Usuario:"))
        self.le_email_user = QLineEdit()
        user_layout.addWidget(self.le_email_user)
        email_form_layout.addLayout(user_layout)

        pass_layout = QHBoxLayout()
        pass_layout.addWidget(QLabel("Contraseña:"))
        self.le_email_pass = QLineEdit()
        self.le_email_pass.setEchoMode(QLineEdit.EchoMode.Password)
        pass_layout.addWidget(self.le_email_pass)
        email_form_layout.addLayout(pass_layout)

        # Recipients
        recipients_layout = QHBoxLayout()
        recipients_layout.addWidget(QLabel("Destinatarios:"))
        self.le_email_to = QLineEdit()
        self.le_email_to.setPlaceholderText("email1@example.com, email2@example.com")
        recipients_layout.addWidget(self.le_email_to)
        email_form_layout.addLayout(recipients_layout)

        # Enable/disable email form
        self.cb_email_enabled.toggled.connect(lambda enabled: self._toggle_email_form(enabled))
        email_layout.addWidget(self.cb_email_enabled)
        email_layout.addLayout(email_form_layout)

        layout.addWidget(email_group)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_save = QPushButton("Guardar")
        self.btn_cancel = QPushButton("Cancelar")
        self.btn_test_email = QPushButton("Probar Email")

        btn_layout.addWidget(self.btn_test_email)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_save)

        layout.addLayout(btn_layout)

        # Connect signals
        self.btn_save.clicked.connect(self._save_config)
        self.btn_cancel.clicked.connect(self.close)
        self.btn_test_email.clicked.connect(self._test_email)

    def _toggle_email_form(self, enabled: bool):
        """Enable/disable email form fields."""
        self.le_smtp_server.setEnabled(enabled)
        self.sb_smtp_port.setEnabled(enabled)
        self.le_email_user.setEnabled(enabled)
        self.le_email_pass.setEnabled(enabled)
        self.le_email_to.setEnabled(enabled)

    def _load_config(self):
        """Load current configuration."""
        config = self.alert_manager.config

        self.cb_enabled.setChecked(config.enabled)
        self.cb_sound.setChecked(config.sound_enabled)
        self.cb_gui.setChecked(config.gui_notifications_enabled)
        self.cb_log.setChecked(config.log_alerts)
        self.sb_auto_ack.setValue(config.auto_acknowledge_after_minutes)

        # Email config
        if hasattr(config, 'email_enabled') and config.email_enabled:
            self.cb_email_enabled.setChecked(True)
            if hasattr(config, 'email_config') and config.email_config:
                email_config = config.email_config
                self.le_smtp_server.setText(email_config.get('server', ''))
                self.sb_smtp_port.setValue(email_config.get('port', 587))
                self.le_email_user.setText(email_config.get('username', ''))
                self.le_email_pass.setText(email_config.get('password', ''))
                self.le_email_to.setText(', '.join(email_config.get('to', [])))

    def _save_config(self):
        """Save configuration."""
        config = AlertConfig(
            enabled=self.cb_enabled.isChecked(),
            sound_enabled=self.cb_sound.isChecked(),
            gui_notifications_enabled=self.cb_gui.isChecked(),
            log_alerts=self.cb_log.isChecked(),
            auto_acknowledge_after_minutes=self.sb_auto_ack.value()
        )

        # Email config
        if self.cb_email_enabled.isChecked():
            config.email_enabled = True
            config.email_config = {
                'server': self.le_smtp_server.text(),
                'port': self.sb_smtp_port.value(),
                'username': self.le_email_user.text(),
                'password': self.le_email_pass.text(),
                'to': [email.strip() for email in self.le_email_to.text().split(',') if email.strip()]
            }

        self.alert_manager.update_config(config)
        QMessageBox.information(self, "Guardado", "Configuración guardada correctamente.")
        self.close()

    def _test_email(self):
        """Test email configuration."""
        if not self.cb_email_enabled.isChecked():
            QMessageBox.warning(self, "Error", "Las notificaciones por email no están habilitadas.")
            return

        # Create test alert
        test_config = {
            'email_server': self.le_smtp_server.text(),
            'email_port': self.sb_smtp_port.value(),
            'email_username': self.le_email_user.text(),
            'email_password': self.le_email_pass.text(),
            'email_to': [email.strip() for email in self.le_email_to.text().split(',') if email.strip()]
        }

        from .notification_handler import NotificationHandler
        from .alert_types import Alert, AlertType, AlertSeverity

        handler = NotificationHandler(test_config)
        test_alert = Alert(
            id="test",
            type=AlertType.CUSTOM,
            severity=AlertSeverity.LOW,
            title="Prueba de Email",
            message="Esta es una alerta de prueba para verificar la configuración de email.",
            timestamp=datetime.now(),
            source="test"
        )

        try:
            handler._notify_email(test_alert)
            QMessageBox.information(self, "Éxito", "Email de prueba enviado correctamente.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al enviar email de prueba: {str(e)}")


# Import here to avoid circular imports
if PYQT6_AVAILABLE:
    from PyQt6.QtWidgets import QLineEdit