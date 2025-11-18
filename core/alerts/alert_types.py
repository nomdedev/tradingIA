"""
Alert types and data structures for the trading platform.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional


class AlertType(Enum):
    """Types of alerts that can be triggered."""
    TRADING_SIGNAL = "trading_signal"
    PERFORMANCE_THRESHOLD = "performance_threshold"
    SYSTEM_STATUS = "system_status"
    STRATEGY_ERROR = "strategy_error"
    DATA_ISSUE = "data_issue"
    CONNECTION_LOST = "connection_lost"
    CONNECTION_RESTORED = "connection_restored"
    CUSTOM = "custom"


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class NotificationMethod(Enum):
    """Methods for delivering notifications."""
    GUI = "gui"
    SOUND = "sound"
    EMAIL = "email"
    LOG = "log"


@dataclass
class Alert:
    """Represents a single alert."""
    id: str
    type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str  # e.g., "strategy_name", "system", "data_manager"
    data: Optional[Dict[str, Any]] = None
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    auto_acknowledge: bool = False
    persistent: bool = False  # If true, alert stays until manually dismissed


@dataclass
class AlertRule:
    """Represents a rule for triggering alerts."""
    id: str
    name: str
    type: AlertType
    enabled: bool = True
    conditions: Dict[str, Any] = None
    notification_methods: list[NotificationMethod] = None
    cooldown_minutes: int = 0  # Minimum time between alerts of this type
    last_triggered: Optional[datetime] = None

    def __post_init__(self):
        if self.conditions is None:
            self.conditions = {}
        if self.notification_methods is None:
            self.notification_methods = [NotificationMethod.GUI]


@dataclass
class AlertConfig:
    """Configuration for the alert system."""
    enabled: bool = True
    max_alerts_history: int = 1000
    auto_acknowledge_after_minutes: int = 0  # 0 = never auto-acknowledge
    sound_enabled: bool = True
    email_enabled: bool = False
    email_config: Optional[Dict[str, str]] = None
    gui_notifications_enabled: bool = True
    log_alerts: bool = True