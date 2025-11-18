"""
Alerts and notifications system for the trading platform.
"""

from .alert_types import (
    Alert,
    AlertRule,
    AlertConfig,
    AlertType,
    AlertSeverity,
    NotificationMethod
)
from .alert_manager import AlertManager
from .notification_handler import NotificationHandler

try:
    from .alerts_panel import AlertsPanel, AlertConfigDialog
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    AlertsPanel = None
    AlertConfigDialog = None

__all__ = [
    'Alert',
    'AlertRule',
    'AlertConfig',
    'AlertType',
    'AlertSeverity',
    'NotificationMethod',
    'AlertManager',
    'NotificationHandler',
]

if PYQT6_AVAILABLE:
    __all__.extend(['AlertsPanel', 'AlertConfigDialog'])