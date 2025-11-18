"""
Alert manager for handling alert rules, triggering, and history.
"""

import logging
import threading
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .alert_types import Alert, AlertRule, AlertConfig, AlertType, AlertSeverity, NotificationMethod
from .notification_handler import NotificationHandler


class AlertManager:
    """Manages alerts, rules, and notifications."""

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.logger = logging.getLogger(__name__)

        # Alert storage
        self.alerts_history: List[Alert] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_rules: Dict[str, AlertRule] = {}

        # Notification handler
        self.notification_handler = NotificationHandler(self._get_notification_config())

        # Threading
        self.lock = threading.RLock()
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None

        # Auto-acknowledge timer
        self.auto_acknowledge_timer: Optional[threading.Timer] = None

    def _get_notification_config(self) -> Dict:
        """Get notification configuration from alert config."""
        return {
            'sound_enabled': self.config.sound_enabled,
            'email_server': getattr(self.config, 'email_server', None),
            'email_port': getattr(self.config, 'email_port', 587),
            'email_username': getattr(self.config, 'email_username', None),
            'email_password': getattr(self.config, 'email_password', None),
            'email_from': getattr(self.config, 'email_from', None),
            'email_to': getattr(self.config, 'email_to', []),
        }

    def start_monitoring(self):
        """Start the alert monitoring system."""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

        if self.config.auto_acknowledge_after_minutes > 0:
            self._start_auto_acknowledge_timer()

        self.logger.info("Alert monitoring started")

    def stop_monitoring(self):
        """Stop the alert monitoring system."""
        self.monitoring_active = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)

        if self.auto_acknowledge_timer:
            self.auto_acknowledge_timer.cancel()

        self.logger.info("Alert monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop for auto-acknowledging alerts."""
        while self.monitoring_active:
            try:
                self._check_auto_acknowledge()
                threading.Event().wait(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Error in alert monitoring loop: {e}")

    def _check_auto_acknowledge(self):
        """Check for alerts that should be auto-acknowledged."""
        if self.config.auto_acknowledge_after_minutes <= 0:
            return

        cutoff_time = datetime.now() - timedelta(minutes=self.config.auto_acknowledge_after_minutes)

        with self.lock:
            alerts_to_ack = []
            for alert in self.active_alerts.values():
                if not alert.acknowledged and alert.timestamp < cutoff_time:
                    alerts_to_ack.append(alert.id)

            for alert_id in alerts_to_ack:
                self.acknowledge_alert(alert_id, auto=True)

    def _start_auto_acknowledge_timer(self):
        """Start the auto-acknowledge timer."""
        def auto_acknowledge_worker():
            while self.monitoring_active:
                self._check_auto_acknowledge()
                threading.Event().wait(self.config.auto_acknowledge_after_minutes * 60)

        self.auto_acknowledge_timer = threading.Timer(0, auto_acknowledge_worker)
        self.auto_acknowledge_timer.daemon = True
        self.auto_acknowledge_timer.start()

    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.id] = rule
            self.logger.info(f"Added alert rule: {rule.name} ({rule.id})")

    def remove_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self.lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                self.logger.info(f"Removed alert rule: {rule_id}")

    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        with self.lock:
            return list(self.alert_rules.values())

    def trigger_alert(self, type: AlertType, severity: AlertSeverity, title: str, message: str,
                     source: str, data: Optional[Dict] = None, persistent: bool = False) -> Optional[str]:
        """Trigger a new alert."""
        if not self.config.enabled:
            return None

        alert_id = str(uuid.uuid4())
        alert = Alert(
            id=alert_id,
            type=type,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            data=data,
            persistent=persistent
        )

        with self.lock:
            # Check if we should trigger based on rules
            if not self._should_trigger_alert(alert):
                return None

            # Add to active alerts
            self.active_alerts[alert_id] = alert

            # Add to history
            self.alerts_history.append(alert)
            if len(self.alerts_history) > self.config.max_alerts_history:
                self.alerts_history.pop(0)

        # Send notifications
        notification_methods = self._get_notification_methods_for_alert(alert)
        self.notification_handler.notify(alert, notification_methods)

        self.logger.info(f"Alert triggered: {alert.title} (ID: {alert_id})")
        return alert_id

    def _should_trigger_alert(self, alert: Alert) -> bool:
        """Check if an alert should be triggered based on rules."""
        # Find matching rules
        matching_rules = []
        for rule in self.alert_rules.values():
            if rule.enabled and rule.type == alert.type:
                if self._check_rule_conditions(rule, alert):
                    matching_rules.append(rule)

        if not matching_rules:
            # No rules match, trigger by default for critical alerts
            return alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]

        # Check cooldown for each matching rule
        for rule in matching_rules:
            if rule.cooldown_minutes > 0 and rule.last_triggered:
                cooldown_end = rule.last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue  # Still in cooldown
            rule.last_triggered = datetime.now()
            return True

        return False

    def _check_rule_conditions(self, rule: AlertRule, alert: Alert) -> bool:
        """Check if alert matches rule conditions."""
        # Simple condition checking - can be extended
        for key, value in rule.conditions.items():
            if key == 'severity' and alert.severity.value != value:
                return False
            elif key == 'source' and alert.source != value:
                return False
            elif key == 'min_severity':
                min_severity_level = ['low', 'medium', 'high', 'critical'].index(value)
                alert_severity_level = ['low', 'medium', 'high', 'critical'].index(alert.severity.value)
                if alert_severity_level < min_severity_level:
                    return False
        return True

    def _get_notification_methods_for_alert(self, alert: Alert) -> List[NotificationMethod]:
        """Get notification methods for an alert."""
        methods = []

        # Check rules first
        for rule in self.alert_rules.values():
            if rule.enabled and rule.type == alert.type and self._check_rule_conditions(rule, alert):
                methods.extend(rule.notification_methods)
                break

        # Default methods if no rules match
        if not methods:
            methods = [NotificationMethod.GUI, NotificationMethod.LOG]
            if alert.severity in [AlertSeverity.HIGH, AlertSeverity.CRITICAL]:
                if self.config.sound_enabled:
                    methods.append(NotificationMethod.SOUND)

        return list(set(methods))  # Remove duplicates

    def acknowledge_alert(self, alert_id: str, auto: bool = False):
        """Acknowledge an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.acknowledged = True
                alert.acknowledged_at = datetime.now()
                alert.auto_acknowledge = auto

                if not alert.persistent:
                    del self.active_alerts[alert_id]

                self.logger.info(f"Alert acknowledged: {alert.title} (ID: {alert_id}, auto: {auto})")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history."""
        with self.lock:
            alerts = self.alerts_history.copy()
            if limit:
                alerts = alerts[-limit:]
            return alerts

    def clear_alert_history(self):
        """Clear alert history."""
        with self.lock:
            self.alerts_history.clear()
            self.logger.info("Alert history cleared")

    def update_config(self, config: AlertConfig):
        """Update alert configuration."""
        with self.lock:
            old_config = self.config
            self.config = config

            # Restart monitoring if needed
            if old_config.enabled != config.enabled:
                if config.enabled:
                    self.start_monitoring()
                else:
                    self.stop_monitoring()

            # Update notification handler config
            self.notification_handler = NotificationHandler(self._get_notification_config())

            self.logger.info("Alert configuration updated")