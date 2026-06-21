"""
TradingIA Platform - Enhanced Dashboard
Improved dashboard with better UX, onboarding hints, and quick actions

Author: TradingIA Team
Version: 2.1.0
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGridLayout,
    QGroupBox,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor
from datetime import datetime
from src.gui.styles import DarkTheme


class WelcomeBanner(QFrame):
    """Welcome banner for new users"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        
        self.setStyleSheet(
            """
            QFrame {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #667eea, stop:1 #764ba2);
                border-radius: 12px;
                color: white;
            }
            QLabel {
                background-color: transparent;
            }
        """
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(20, 15, 20, 15)

        # Welcome text
        welcome_text = QWidget()
        welcome_layout = QVBoxLayout(welcome_text)

        title = QLabel("🎉 ¡Bienvenido a TradingIA!")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        welcome_layout.addWidget(title)

        subtitle = QLabel("La plataforma más avanzada para trading algorítmico")
        subtitle.setFont(QFont("Arial", 11))
        subtitle.setStyleSheet("color: rgba(255,255,255,0.9);")
        welcome_layout.addWidget(subtitle)

        layout.addWidget(welcome_text)

        # Quick start button
        self.quick_start_btn = QPushButton("🚀 Comenzar Tutorial")
        self.quick_start_btn.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        self.quick_start_btn.setStyleSheet(
            """
            QPushButton {
                background-color: rgba(255,255,255,0.2);
                color: white;
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgba(255,255,255,0.3);
                border-color: rgba(255,255,255,0.5);
            }
        """
        )
        layout.addStretch()
        layout.addWidget(self.quick_start_btn)

        self.setLayout(layout)


class OnboardingCard(QFrame):
    """Onboarding progress card"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # No fixed height, but let it wrap content tightly
        
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {DarkTheme.BG_SECONDARY};
                border-radius: 8px;
                border: 1px solid {DarkTheme.BORDER_COLOR};
            }}
            QLabel {{
                background-color: transparent;
                border: none;
            }}
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("📋 Progreso de Configuración")
        title.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {DarkTheme.TEXT_HIGHLIGHT};")
        layout.addWidget(title)

        # Progress items
        self.progress_items = []

        steps = [
            ("Cargar Datos", "load_data", False),
            ("Seleccionar Estrategia", "select_strategy", False),
            ("Ejecutar Backtest", "run_backtest", False),
            ("Configurar Riesgos", "setup_risk", False),
            ("Listo para Trading", "ready_trading", False),
        ]

        for step_name, step_id, completed in steps:
            item_layout = QHBoxLayout()

            checkbox = QLabel("❌" if not completed else "✅")
            checkbox.setStyleSheet(f"color: {DarkTheme.ERROR};" if not completed else f"color: {DarkTheme.SUCCESS};")
            item_layout.addWidget(checkbox)

            step_label = QLabel(step_name)
            step_label.setStyleSheet(f"color: {DarkTheme.TEXT_PRIMARY};")
            item_layout.addWidget(step_label)

            item_layout.addStretch()

            # Action button for incomplete steps
            if not completed:
                action_btn = QPushButton("Ir")
                action_btn.setFixedWidth(40)
                action_btn.setStyleSheet(
                    f"""
                    QPushButton {{
                        background-color: {DarkTheme.ACCENT_PRIMARY};
                        color: {DarkTheme.TEXT_HIGHLIGHT};
                        border: none;
                        border-radius: 4px;
                        padding: 2px;
                    }}
                    QPushButton:hover {{
                        background-color: {DarkTheme.ACCENT_HOVER};
                    }}
                """
                )
                action_btn.setProperty("step_id", step_id)
                item_layout.addWidget(action_btn)

            layout.addLayout(item_layout)
            self.progress_items.append((step_id, checkbox, step_label))

        self.setLayout(layout)

    def update_progress(self, step_id, completed):
        """Update progress for a specific step"""
        for item_id, checkbox, label in self.progress_items:
            if item_id == step_id:
                if completed:
                    checkbox.setText("✅")
                    checkbox.setStyleSheet(f"color: {DarkTheme.SUCCESS};")
                    label.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; text-decoration: line-through;")
                else:
                    checkbox.setText("❌")
                    checkbox.setStyleSheet(f"color: {DarkTheme.ERROR};")
                    label.setStyleSheet(f"color: {DarkTheme.TEXT_PRIMARY};")
                break


class EnhancedMetricCard(QFrame):
    """Enhanced metric display card with trend indicators"""

    def __init__(self, title, value, subtitle="", trend=None, color="#0e639c"):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(100)
        self.setMaximumHeight(120)
        
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {color}22;
                border: 1px solid {color}44;
                border-radius: 8px;
                color: {DarkTheme.TEXT_PRIMARY};
            }}
            QLabel {{
                background-color: transparent;
                border: none;
            }}
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(12, 12, 12, 12)

        # Title with trend indicator
        title_layout = QHBoxLayout()

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        title_layout.addWidget(title_label)

        if trend is not None:
            trend_icon = "📈" if trend > 0 else "📉" if trend < 0 else "➡️"
            trend_color = DarkTheme.SUCCESS if trend > 0 else DarkTheme.ERROR if trend < 0 else DarkTheme.TEXT_SECONDARY
            trend_label = QLabel(trend_icon)
            trend_label.setStyleSheet(f"color: {trend_color}; font-size: 13px;")
            title_layout.addWidget(trend_label)

        title_layout.addStretch()
        layout.addLayout(title_layout)

        # Value
        self.value_label = QLabel(str(value))
        self.value_label.setFont(QFont("Arial", 18, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {color};")
        layout.addWidget(self.value_label)

        # Subtitle
        if subtitle:
            self.subtitle_label = QLabel(subtitle)
            self.subtitle_label.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 11px;")
            layout.addWidget(self.subtitle_label)

        self.setLayout(layout)

    def update_value(self, value, trend=None):
        """Update metric value and trend"""
        self.value_label.setText(str(value))
        # Could update trend indicator here


class QuickActionCard(QFrame):
    """Quick action card with description"""

    def __init__(self, icon, title, description, button_text, color="#4299e1", parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(160)
        self.setMaximumHeight(180)

        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: #2d3748;
                border: 1px solid #4a5568;
                border-radius: 8px;
                color: #e2e8f0;
            }}
            QFrame:hover {{
                border-color: {color};
                background-color: #374151;
            }}
            QLabel {{
                background-color: transparent;
                border: none;
            }}
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        # Icon and title
        header_layout = QHBoxLayout()

        icon_label = QLabel(icon)
        icon_label.setFont(QFont("Arial", 20))
        header_layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setFont(QFont("Arial", 13, QFont.Weight.Bold))
        header_layout.addWidget(title_label)

        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Description
        desc_label = QLabel(description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #a0aec0; font-size: 11px; margin-top: 4px;")
        layout.addWidget(desc_label)

        # Button
        self.action_button = QPushButton(button_text)
        self.action_button.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background-color: {self._darken_color(color)};
            }}
            QPushButton:pressed {{
                background-color: {self._darken_color(self._darken_color(color))};
            }}
        """
        )
        layout.addWidget(self.action_button)

        self.setLayout(layout)

    def _darken_color(self, hex_color):
        """Simple color darkening"""
        # This is a simplified implementation
        if hex_color == "#4299e1":
            return "#3182ce"
        elif hex_color == "#3182ce":
            return "#2c5282"
        return hex_color


class EnhancedTab0Dashboard(QWidget):
    """Enhanced dashboard with better UX and onboarding"""

    # Signals
    tutorial_requested = Signal()
    load_data_clicked = Signal()
    select_strategy_clicked = Signal()
    run_backtest_clicked = Signal()
    setup_risk_clicked = Signal()
    start_live_clicked = Signal()
    help_requested = Signal()

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform

        # State tracking
        self.onboarding_progress = {
            "load_data": False,
            "select_strategy": False,
            "run_backtest": False,
            "setup_risk": False,
            "ready_trading": False,
        }

        self.init_ui()
        self.connect_signals()
        self.update_onboarding_status()

    def init_ui(self):
        """Initialize the enhanced dashboard UI"""
        # Main layout for the tab (horizontal to allow centering)
        tab_layout = QHBoxLayout()
        tab_layout.setContentsMargins(0, 0, 0, 0)

        # Center spacer
        tab_layout.addStretch(1)

        # Content widget to hold the dashboard elements
        content_widget = QWidget()
        content_widget.setMaximumWidth(1600)  # Increased width for better fullscreen usage

        # Vertical layout for the content
        main_layout = QVBoxLayout(content_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Welcome banner for new users
        self.welcome_banner = WelcomeBanner()
        self.welcome_banner.quick_start_btn.clicked.connect(self.tutorial_requested.emit)
        main_layout.addWidget(self.welcome_banner)

        # Onboarding progress
        self.onboarding_card = OnboardingCard()
        main_layout.addWidget(self.onboarding_card)

        # Metrics overview
        metrics_group = self.create_metrics_section()
        main_layout.addWidget(metrics_group)

        # Quick actions grid
        actions_group = self.create_quick_actions()
        main_layout.addWidget(actions_group)

        # Recent activity
        activity_group = self.create_activity_section()
        main_layout.addWidget(activity_group)

        main_layout.addStretch()

        # Add content widget to tab layout
        tab_layout.addWidget(content_widget, 4)  # Give it more weight

        # Center spacer
        tab_layout.addStretch(1)

        self.setLayout(tab_layout)

    def create_metrics_section(self):
        """Create enhanced metrics overview"""
        group = QGroupBox("📊 Estado del Sistema")
        # Remove fixed size policy to allow expansion
        group.setStyleSheet(
            """
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #e2e8f0;
                border: 2px solid #4a5568;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: #1a202c;
            }
        """
        )

        layout = QGridLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)
        
        # System status metrics
        self.data_status_card = EnhancedMetricCard("Datos", "No cargados", "Carga datos para comenzar", color="#e53e3e")

        self.strategy_status_card = EnhancedMetricCard(
            "Estrategia", "No seleccionada", "Elige una estrategia", color="#d69e2e"
        )

        self.backtest_status_card = EnhancedMetricCard(
            "Backtest", "No ejecutado", "Prueba tu estrategia", color="#38a169"
        )

        self.live_status_card = EnhancedMetricCard("Live Trading", "Detenido", "Comienza a operar", color="#3182ce")

        # Use 4 columns for better horizontal distribution
        layout.addWidget(self.data_status_card, 0, 0)
        layout.addWidget(self.strategy_status_card, 0, 1)
        layout.addWidget(self.backtest_status_card, 0, 2)
        layout.addWidget(self.live_status_card, 0, 3)

        group.setLayout(layout)
        return group

    def create_quick_actions(self):
        """Create quick actions grid"""
        group = QGroupBox("⚡ Acciones Rápidas")
        # Remove fixed size policy to allow expansion
        group.setStyleSheet(
            """
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #e2e8f0;
                border: 2px solid #4a5568;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: #1a202c;
            }
        """
        )

        layout = QGridLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(15, 20, 15, 15)
        
        # Quick action cards
        self.data_action = QuickActionCard(
            "📊", "Cargar Datos", "Importa datos históricos", "Cargar", "#4299e1"
        )
        self.data_action.action_button.clicked.connect(self.on_data_action_clicked)

        self.strategy_action = QuickActionCard(
            "🎯", "Seleccionar Estrategia", "Elige una estrategia", "Seleccionar", "#48bb78"
        )
        self.strategy_action.action_button.clicked.connect(self.on_strategy_action_clicked)

        self.backtest_action = QuickActionCard(
            "▶️", "Ejecutar Backtest", "Prueba tu estrategia", "Ejecutar", "#ed8936"
        )
        self.backtest_action.action_button.clicked.connect(self.on_backtest_action_clicked)

        self.live_action = QuickActionCard(
            "🔴", "Iniciar Live Trading", "Comienza a operar", "Iniciar", "#e53e3e"
        )
        self.live_action.action_button.clicked.connect(self.on_live_action_clicked)

        # Use 4 columns for better horizontal distribution
        layout.addWidget(self.data_action, 0, 0)
        layout.addWidget(self.strategy_action, 0, 1)
        layout.addWidget(self.backtest_action, 0, 2)
        layout.addWidget(self.live_action, 0, 3)

        group.setLayout(layout)
        return group

    # Debug slots to ensure signals are firing
    def on_data_action_clicked(self):
        print("DEBUG: Data action clicked")
        self.load_data_clicked.emit()

    def on_strategy_action_clicked(self):
        print("DEBUG: Strategy action clicked")
        self.select_strategy_clicked.emit()

    def on_backtest_action_clicked(self):
        print("DEBUG: Backtest action clicked")
        self.run_backtest_clicked.emit()

    def on_live_action_clicked(self):
        print("DEBUG: Live action clicked")
        self.start_live_clicked.emit()

    def create_activity_section(self):
        """Create recent activity section"""
        group = QGroupBox("📝 Actividad Reciente")
        group.setStyleSheet(
            """
            QGroupBox {
                font-size: 14px;
                font-weight: bold;
                color: #e2e8f0;
                border: 2px solid #4a5568;
                border-radius: 8px;
                margin-top: 8px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 12px;
                padding: 0 8px;
                background-color: #1a202c;
            }
        """
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 20, 15, 15)

        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet(
            """
            QListWidget {
                background-color: #2d3748;
                border: 1px solid #4a5568;
                border-radius: 4px;
                color: #e2e8f0;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #4a5568;
            }
            QListWidget::item:hover {
                background-color: #374151;
            }
        """
        )

        # Add some initial activity items
        self.add_activity_item("🕐 Aplicación iniciada", "sistema")
        self.add_activity_item("ℹ️ Esperando configuración inicial", "sistema")

        layout.addWidget(self.activity_list)
        group.setLayout(layout)
        return group

    def connect_signals(self):
        """Connect internal signals"""
        # Connect onboarding card action buttons
        for item_layout in self.onboarding_card.findChildren(QHBoxLayout):
            for widget in item_layout.children():
                if isinstance(widget, QPushButton) and widget.property("step_id"):
                    step_id = widget.property("step_id")
                    if step_id == "load_data":
                        widget.clicked.connect(self.load_data_clicked.emit)
                    elif step_id == "select_strategy":
                        widget.clicked.connect(self.select_strategy_clicked.emit)
                    elif step_id == "run_backtest":
                        widget.clicked.connect(self.run_backtest_clicked.emit)
                    elif step_id == "setup_risk":
                        widget.clicked.connect(self.setup_risk_clicked.emit)

    def update_onboarding_status(self):
        """Update onboarding progress display"""
        for step_id, completed in self.onboarding_progress.items():
            self.onboarding_card.update_progress(step_id, completed)

    def update_system_status(self, component, status):
        """Update system component status"""
        status_colors = {
            "data": {"loaded": "#48bb78", "not_loaded": "#e53e3e"},
            "strategy": {"selected": "#48bb78", "not_selected": "#d69e2e"},
            "backtest": {"completed": "#48bb78", "not_run": "#38a169"},
            "live": {"running": "#48bb78", "stopped": "#3182ce"},
        }

        if component == "data":
            color = status_colors["data"].get(status, "#e53e3e")
            text = "Cargados" if status == "loaded" else "No cargados"
            self.data_status_card.value_label.setText(text)
            self.data_status_card.value_label.setStyleSheet(f"color: {color};")
            self.onboarding_progress["load_data"] = status == "loaded"

        elif component == "strategy":
            color = status_colors["strategy"].get(status, "#d69e2e")
            text = "Seleccionada" if status == "selected" else "No seleccionada"
            self.strategy_status_card.value_label.setText(text)
            self.strategy_status_card.value_label.setStyleSheet(f"color: {color};")
            self.onboarding_progress["select_strategy"] = status == "selected"

        elif component == "backtest":
            color = status_colors["backtest"].get(status, "#38a169")
            text = "Completado" if status == "completed" else "No ejecutado"
            self.backtest_status_card.value_label.setText(text)
            self.backtest_status_card.value_label.setStyleSheet(f"color: {color};")
            self.onboarding_progress["run_backtest"] = status == "completed"

        elif component == "live":
            color = status_colors["live"].get(status, "#3182ce")
            text = "Ejecutándose" if status == "running" else "Detenido"
            self.live_status_card.value_label.setText(text)
            self.live_status_card.value_label.setStyleSheet(f"color: {color};")
            self.onboarding_progress["ready_trading"] = status == "running"

        self.update_onboarding_status()

    def add_activity_item(self, message, category="user"):
        """Add an activity item to the list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}"

        item = QListWidgetItem(full_message)
        item.setToolTip(f"Categoría: {category}")

        # Color coding by category
        if category == "error":
            item.setForeground(QColor("#fc8181"))
        elif category == "success":
            item.setForeground(QColor("#68d391"))
        elif category == "warning":
            item.setForeground(QColor("#ed8936"))
        else:
            item.setForeground(QColor("#e2e8f0"))

        self.activity_list.addItem(item)
        self.activity_list.scrollToBottom()

    def hide_welcome_banner(self):
        """Hide the welcome banner (user has seen it)"""
        self.welcome_banner.hide()

    def show_help_tooltip(self, widget, message):
        """Show contextual help for a widget"""
        widget.setToolTip(message)
