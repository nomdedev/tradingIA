"""
TradingIA Platform - Tab 0: Dashboard
Main overview and quick actions

Author: TradingIA Team
Version: 2.0.0
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QGridLayout, QGroupBox, QListWidget, QListWidgetItem
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QFont, QColor
from datetime import datetime


class MetricCard(QFrame):
    """Modern metric display card"""
    
    def __init__(self, title, value, subtitle="", color="#0e639c"):
        super().__init__()
        self.color = color
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: #2d2d2d;
                border: 2px solid {color};
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(8)
        
        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #888; font-size: 12px; font-weight: 600;")
        layout.addWidget(title_label)
        
        # Value
        self.value_label = QLabel(str(value))
        self.value_label.setStyleSheet(f"""
            color: {color};
            font-size: 32px;
            font-weight: 700;
        """)
        layout.addWidget(self.value_label)
        
        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet("color: #666; font-size: 11px;")
        layout.addWidget(self.subtitle_label)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def update_value(self, value):
        """Update metric value"""
        self.value_label.setText(str(value))
    
    def update_subtitle(self, subtitle):
        """Update subtitle text"""
        self.subtitle_label.setText(subtitle)


class QuickActionButton(QPushButton):
    """Styled quick action button"""
    
    def __init__(self, icon, text, color="#0e639c"):
        super().__init__(f"{icon}  {text}")
        self.setFixedHeight(60)
        self.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {color}, stop:1 {self._darken_color(color)});
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 14px;
                font-weight: 600;
                text-align: left;
                padding: 12px 20px;
            }}
            QPushButton:hover {{
                background: {self._lighten_color(color)};
            }}
            QPushButton:pressed {{
                background: {self._darken_color(color)};
            }}
        """)
    
    def _darken_color(self, hex_color):
        """Darken a hex color"""
        # Simple darkening - reduce RGB values
        return hex_color  # Simplified for now
    
    def _lighten_color(self, hex_color):
        """Lighten a hex color"""
        return hex_color  # Simplified for now


class Tab0Dashboard(QWidget):
    """Main dashboard tab with overview and quick actions"""
    
    # Signals for quick actions
    load_data_clicked = Signal()
    run_backtest_clicked = Signal()
    start_live_clicked = Signal()
    open_strategy_clicked = Signal()
    
    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        
        # Initial state - no data yet
        self.current_balance = 0.00
        self.current_pnl = 0.00
        self.win_rate = 0.00
        self.active_trades = 0
        self.has_data = False
        self.last_backtest_results = None
        
        self.init_ui()
        # Don't start updates automatically - will start when needed
    
    def init_ui(self):
        """Initialize dashboard UI"""
        main_layout = QVBoxLayout()
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # Header with status
        header_layout = QHBoxLayout()
        header = QLabel("📊 Trading Dashboard")
        header.setStyleSheet("""
            font-size: 28px;
            font-weight: 700;
            color: #fff;
            margin-bottom: 10px;
        """)
        header_layout.addWidget(header)
        
        # Status indicator
        self.status_label = QLabel("⚪ No Data Loaded")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #888;
            padding: 8px 16px;
            background-color: #2d2d2d;
            border-radius: 6px;
        """)
        header_layout.addWidget(self.status_label)
        header_layout.addStretch()
        
        main_layout.addLayout(header_layout)
        
        # Info banner
        info_banner = QLabel(
            "💡 <b>Getting Started:</b> Use the quick actions below to load data, "
            "configure a strategy, and run your first backtest. "
            "Results will appear here once you start trading or backtesting."
        )
        info_banner.setWordWrap(True)
        info_banner.setStyleSheet("""
            QLabel {
                background-color: #1e3a5f;
                color: #a8d8ff;
                padding: 12px 16px;
                border-radius: 8px;
                border-left: 4px solid #569cd6;
                font-size: 13px;
            }
        """)
        main_layout.addWidget(info_banner)
        
        # Metrics Overview
        metrics_group = self.create_metrics_section()
        main_layout.addWidget(metrics_group)
        
        # Quick Actions
        actions_group = self.create_quick_actions()
        main_layout.addWidget(actions_group)
        
        # Recent Activity (will be hidden initially)
        self.activity_group = self.create_activity_section()
        self.activity_group.setVisible(False)
        main_layout.addWidget(self.activity_group)
        
        main_layout.addStretch()
        self.setLayout(main_layout)
    
    def create_metrics_section(self):
        """Create metrics overview section"""
        group = QGroupBox("Portfolio Overview")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: 600;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }
        """)
        
        layout = QGridLayout()
        layout.setSpacing(16)
        layout.setContentsMargins(16, 24, 16, 16)
        
        # Create metric cards with explanatory tooltips
        self.balance_card = MetricCard(
            "Balance",
            "No Data",
            "Load backtest or start trading",
            "#4ec9b0"
        )
        self.balance_card.setToolTip(
            "Shows your current capital. Will update when you:\n"
            "• Run a backtest (shows final backtest balance)\n"
            "• Start live trading (shows real-time balance)"
        )
        
        self.pnl_card = MetricCard(
            "P&L",
            "No Data",
            "Run backtest to see results",
            "#569cd6"
        )
        self.pnl_card.setToolTip(
            "Profit & Loss tracking. Shows:\n"
            "• Backtest: Total P&L from simulation\n"
            "• Live: Real-time P&L from open positions"
        )
        
        self.winrate_card = MetricCard(
            "Win Rate",
            "No Data",
            "Execute trades to calculate",
            "#dcdcaa"
        )
        self.winrate_card.setToolTip(
            "Percentage of profitable trades.\n"
            "Requires at least 10 completed trades to display."
        )
        
        self.trades_card = MetricCard(
            "Trades",
            "0",
            "No active positions",
            "#c586c0"
        )
        self.trades_card.setToolTip(
            "Number of active trades:\n"
            "• Backtest: Total trades executed\n"
            "• Live: Currently open positions"
        )
        
        # Add cards to grid
        layout.addWidget(self.balance_card, 0, 0)
        layout.addWidget(self.pnl_card, 0, 1)
        layout.addWidget(self.winrate_card, 0, 2)
        layout.addWidget(self.trades_card, 0, 3)
        
        group.setLayout(layout)
        return group
    
    def create_quick_actions(self):
        """Create quick actions section"""
        group = QGroupBox("Quick Actions")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: 600;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }
        """)
        
        layout = QHBoxLayout()
        layout.setSpacing(12)
        layout.setContentsMargins(16, 24, 16, 16)
        
        # Action buttons
        load_btn = QuickActionButton("📥", "Load Market Data", "#0e639c")
        load_btn.clicked.connect(self.on_load_data)
        
        strategy_btn = QuickActionButton("⚙️", "Configure Strategy", "#c586c0")
        strategy_btn.clicked.connect(self.on_open_strategy)
        
        backtest_btn = QuickActionButton("▶️", "Run Backtest", "#4ec9b0")
        backtest_btn.clicked.connect(self.on_run_backtest)
        
        live_btn = QuickActionButton("🔴", "Start Live Trading", "#f48771")
        live_btn.clicked.connect(self.on_start_live)
        
        layout.addWidget(load_btn)
        layout.addWidget(strategy_btn)
        layout.addWidget(backtest_btn)
        layout.addWidget(live_btn)
        
        group.setLayout(layout)
        return group
    
    def create_activity_section(self):
        """Create recent activity section"""
        group = QGroupBox("Recent Activity")
        group.setStyleSheet("""
            QGroupBox {
                font-size: 16px;
                font-weight: 600;
                color: #fff;
                border: 2px solid #3d3d3d;
                border-radius: 12px;
                margin-top: 12px;
                padding-top: 16px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
            }
        """)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 24, 16, 16)
        
        # Activity list
        self.activity_list = QListWidget()
        self.activity_list.setStyleSheet("""
            QListWidget {
                background-color: #252525;
                border: 1px solid #3d3d3d;
                border-radius: 8px;
                color: #ccc;
                font-size: 13px;
                padding: 8px;
            }
            QListWidget::item {
                padding: 12px;
                border-bottom: 1px solid #3d3d3d;
            }
            QListWidget::item:hover {
                background-color: #2d2d2d;
            }
        """)
        
        # Add sample activities
        self.add_activity("🟢", "System started successfully", "info")
        self.add_activity("📊", "Ready to load market data", "info")
        
        layout.addWidget(self.activity_list)
        group.setLayout(layout)
        return group
    
    def add_activity(self, icon, message, activity_type="info"):
        """Add activity to the list"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        colors = {
            "success": "#4ec9b0",
            "warning": "#dcdcaa",
            "error": "#f48771",
            "info": "#569cd6"
        }
        
        color = colors.get(activity_type, "#569cd6")
        
        item_text = f"{icon} [{timestamp}] {message}"
        item = QListWidgetItem(item_text)
        item.setForeground(QColor(color))
        
        self.activity_list.insertItem(0, item)
        
        # Keep only last 10 items
        while self.activity_list.count() > 10:
            self.activity_list.takeItem(self.activity_list.count() - 1)
    
    def update_from_backtest(self, results):
        """Update dashboard with backtest results"""
        if not results:
            return
        
        self.has_data = True
        self.last_backtest_results = results
        
        # Update status
        self.status_label.setText("🔵 Backtest Results Loaded")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #4ec9b0;
            padding: 8px 16px;
            background-color: #1e3a2d;
            border-radius: 6px;
        """)
        
        # Update metrics from backtest
        final_balance = results.get('final_balance', 0)
        total_pnl = results.get('total_pnl', 0)
        win_rate = results.get('win_rate', 0)
        total_trades = results.get('total_trades', 0)
        
        self.balance_card.update_value(f"${final_balance:,.2f}")
        self.balance_card.update_subtitle("From last backtest")
        
        pnl_color = "#4ec9b0" if total_pnl >= 0 else "#f48771"
        self.pnl_card.update_value(f"${total_pnl:+,.2f}")
        self.pnl_card.update_subtitle(f"{(total_pnl/10000*100):+.2f}% return")
        self.pnl_card.setStyleSheet(f"QGroupBox {{ border-left: 4px solid {pnl_color}; }}")
        
        self.winrate_card.update_value(f"{win_rate:.1f}%")
        self.winrate_card.update_subtitle(f"Based on {total_trades} trades")
        
        self.trades_card.update_value(str(total_trades))
        self.trades_card.update_subtitle("Total executed")
        
        # Show activity section
        self.activity_group.setVisible(True)
        self.add_activity("✅", f"Backtest completed: {total_trades} trades, {win_rate:.1f}% win rate", "success")
    
    def update_from_live_trading(self, balance, pnl, open_trades):
        """Update dashboard with live trading data"""
        self.has_data = True
        
        # Update status
        self.status_label.setText("🔴 Live Trading Active")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #f48771;
            padding: 8px 16px;
            background-color: #3a1e1e;
            border-radius: 6px;
            animation: pulse 2s infinite;
        """)
        
        # Update metrics
        self.balance_card.update_value(f"${balance:,.2f}")
        self.balance_card.update_subtitle("Live balance")
        
        pnl_color = "#4ec9b0" if pnl >= 0 else "#f48771"
        self.pnl_card.update_value(f"${pnl:+,.2f}")
        self.pnl_card.update_subtitle(f"{(pnl/balance*100):+.2f}% today")
        self.pnl_card.setStyleSheet(f"QGroupBox {{ border-left: 4px solid {pnl_color}; }}")
        
        self.trades_card.update_value(str(open_trades))
        self.trades_card.update_subtitle("Open positions")
        
        # Show activity section
        self.activity_group.setVisible(True)
    
    def clear_data(self):
        """Clear all dashboard data"""
        self.has_data = False
        self.last_backtest_results = None
        
        self.status_label.setText("⚪ No Data Loaded")
        self.status_label.setStyleSheet("""
            font-size: 14px;
            color: #888;
            padding: 8px 16px;
            background-color: #2d2d2d;
            border-radius: 6px;
        """)
        
        self.balance_card.update_value("No Data")
        self.balance_card.update_subtitle("Load backtest or start trading")
        
        self.pnl_card.update_value("No Data")
        self.pnl_card.update_subtitle("Run backtest to see results")
        
        self.winrate_card.update_value("No Data")
        self.winrate_card.update_subtitle("Execute trades to calculate")
        
        self.trades_card.update_value("0")
        self.trades_card.update_subtitle("No active positions")
        
        self.activity_group.setVisible(False)
    
    # Quick action handlers
    def on_load_data(self):
        """Handle load data button"""
        self.add_activity("📥", "Navigating to Data Manager...", "info")
        self.parent_platform.tabs.setCurrentIndex(1)  # Go to Data tab
        self.load_data_clicked.emit()
    
    def on_open_strategy(self):
        """Handle open strategy button"""
        self.add_activity("⚙️", "Opening Strategy Builder...", "info")
        self.parent_platform.tabs.setCurrentIndex(2)  # Go to Strategy tab
        self.open_strategy_clicked.emit()
    
    def on_run_backtest(self):
        """Handle run backtest button"""
        # Check if data is loaded
        if not self.parent_platform.data_dict:
            self.add_activity("⚠️", "Please load data first", "warning")
            return
        
        self.add_activity("▶️", "Starting backtest...", "success")
        self.parent_platform.tabs.setCurrentIndex(3)  # Go to Backtest tab
        self.run_backtest_clicked.emit()
    
    def on_start_live(self):
        """Handle start live button"""
        self.add_activity("🔴", "Opening Live Trading...", "info")
        self.parent_platform.tabs.setCurrentIndex(6)  # Go to Live tab
        self.start_live_clicked.emit()
    
    def on_tab_activated(self):
        """Called when this tab is activated"""
        self.add_activity("👁️", "Dashboard refreshed", "info")
