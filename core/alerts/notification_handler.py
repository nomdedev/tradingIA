"""
Notification handler for delivering alerts through various channels.
"""

import logging
import smtplib
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Dict, Optional

try:
    from PyQt6.QtWidgets import QMessageBox, QApplication
    from PyQt6.QtCore import pyqtSignal, QObject
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    # Create dummy classes for type hints
    class QObject:
        def __init__(self):
            self.gui_notification_signal = None
    def pyqtSignal(*args):
        return None
    QMessageBox = None
    QApplication = None

from .alert_types import Alert, AlertSeverity, NotificationMethod


class NotificationHandler(QObject):
    """Handles delivery of notifications through various methods."""

    # Signal for GUI notifications
    gui_notification_signal = pyqtSignal(Alert)

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Email configuration
        self.email_server = self.config.get('email_server')
        self.email_port = self.config.get('email_port', 587)
        self.email_username = self.config.get('email_username')
        self.email_password = self.config.get('email_password')
        self.email_from = self.config.get('email_from')
        self.email_to = self.config.get('email_to', [])

        # Sound configuration
        self.sound_enabled = self.config.get('sound_enabled', True)
        self.sound_files = {
            AlertSeverity.LOW: self.config.get('sound_file_low'),
            AlertSeverity.MEDIUM: self.config.get('sound_file_medium'),
            AlertSeverity.HIGH: self.config.get('sound_file_high'),
            AlertSeverity.CRITICAL: self.config.get('sound_file_critical'),
        }

    def notify(self, alert: Alert, methods: list[NotificationMethod]):
        """Send notification through specified methods."""
        for method in methods:
            try:
                if method == NotificationMethod.GUI:
                    self._notify_gui(alert)
                elif method == NotificationMethod.SOUND:
                    self._notify_sound(alert)
                elif method == NotificationMethod.EMAIL:
                    self._notify_email(alert)
                elif method == NotificationMethod.LOG:
                    self._notify_log(alert)
            except Exception as e:
                self.logger.error(f"Failed to send {method.value} notification for alert {alert.id}: {e}")

    def _notify_gui(self, alert: Alert):
        """Show GUI notification."""
        if not PYQT6_AVAILABLE:
            self.logger.warning("PyQt6 not available, skipping GUI notification")
            return

        # Emit signal for GUI to handle
        if hasattr(self, 'gui_notification_signal') and self.gui_notification_signal:
            self.gui_notification_signal.emit(alert)

        # Also show immediate message box for critical alerts
        if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
            self._show_message_box(alert)

    def _show_message_box(self, alert: Alert):
        """Show a message box for critical alerts."""
        if not PYQT6_AVAILABLE or not QMessageBox or not QApplication:
            return

        # Get the main application window
        app = QApplication.instance()
        if app is None:
            return

        # Determine icon based on severity
        icon = QMessageBox.Icon.Information
        if alert.severity == AlertSeverity.HIGH:
            icon = QMessageBox.Icon.Warning
        elif alert.severity == AlertSeverity.CRITICAL:
            icon = QMessageBox.Icon.Critical

        # Show message box
        msg_box = QMessageBox()
        msg_box.setIcon(icon)
        msg_box.setWindowTitle(f"Trading Alert - {alert.severity.value.upper()}")
        msg_box.setText(alert.title)
        msg_box.setInformativeText(alert.message)
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Show in a separate thread to avoid blocking
        threading.Thread(target=lambda: msg_box.exec(), daemon=True).start()

    def _notify_sound(self, alert: Alert):
        """Play sound notification."""
        if not self.sound_enabled:
            return

        sound_file = self.sound_files.get(alert.severity)
        if not sound_file or not Path(sound_file).exists():
            # Use default system beep if no sound file
            self._play_system_beep()
            return

        try:
            # Import playsound for sound playback
            from playsound import playsound
            threading.Thread(target=lambda: playsound(sound_file), daemon=True).start()
        except ImportError:
            self.logger.warning("playsound not available, using system beep")
            self._play_system_beep()
        except Exception as e:
            self.logger.error(f"Failed to play sound: {e}")
            self._play_system_beep()

    def _play_system_beep(self):
        """Play system beep sound."""
        try:
            import winsound
            # Different frequencies for different severities
            frequency = {
                AlertSeverity.LOW: 800,
                AlertSeverity.MEDIUM: 1000,
                AlertSeverity.HIGH: 1200,
                AlertSeverity.CRITICAL: 1500,
            }.get(AlertSeverity.MEDIUM, 1000)  # Default to medium

            winsound.Beep(frequency, 500)
        except ImportError:
            # Fallback for non-Windows systems
            print('\a')  # ASCII bell character

    def _notify_email(self, alert: Alert):
        """Send email notification."""
        if not self.email_server or not self.email_to or not self.email_username or not self.email_password:
            self.logger.warning("Email not configured, skipping email notification")
            return

        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.email_from or self.email_username
            msg['To'] = ', '.join(self.email_to)
            msg['Subject'] = f"Trading Alert: {alert.title}"

            body = f"""
Trading Alert - {alert.severity.value.upper()}

Title: {alert.title}
Message: {alert.message}
Source: {alert.source}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Severity: {alert.severity.value}
Type: {alert.type.value}
"""

            if alert.data:
                body += f"\nAdditional Data:\n{alert.data}"

            msg.attach(MIMEText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(self.email_server, self.email_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            text = msg.as_string()
            server.sendmail(self.email_from or self.email_username, self.email_to, text)
            server.quit()

            self.logger.info(f"Email notification sent for alert {alert.id}")

        except Exception as e:
            self.logger.error(f"Failed to send email notification: {e}")

    def _notify_log(self, alert: Alert):
        """Log the alert."""
        log_message = f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message} (Source: {alert.source})"

        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif alert.severity == AlertSeverity.HIGH:
            self.logger.error(log_message)
        elif alert.severity == AlertSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)