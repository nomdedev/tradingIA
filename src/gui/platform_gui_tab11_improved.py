import sys
import os
import numpy as np
import pandas as pd
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGroupBox,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QProgressBar,
    QComboBox,
    QPushButton,
    QTextEdit,
    QSplitter,
    QFrame,
    QScrollArea,
    QSizePolicy,
    QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QFont, QPalette, QColor
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from src.gui.styles import DarkTheme

# Import Risk Calculator
try:
    from core.risk.risk_metrics import RiskMetricsCalculator
except ImportError:
    # Fallback for development/testing if module not found in path
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from core.risk.risk_metrics import RiskMetricsCalculator


class MetricCard(QFrame):
    """Reusable Card Component for Risk Metrics"""
    def __init__(self, title, suffix="", tooltip="", parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Raised)
        self.setToolTip(tooltip)
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: {DarkTheme.BG_SECONDARY};
                border: 1px solid {DarkTheme.BORDER_COLOR};
                border-radius: 8px;
            }}
            MetricCard:hover {{
                border: 1px solid {DarkTheme.ACCENT_PRIMARY};
                background-color: {DarkTheme.BG_TERTIARY};
            }}
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(5)
        
        # Title
        self.title_lbl = QLabel(title.upper())
        self.title_lbl.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 11px; font-weight: 600; letter-spacing: 0.5px;")
        layout.addWidget(self.title_lbl)
        
        # Value Row
        val_layout = QHBoxLayout()
        val_layout.setSpacing(4)
        val_layout.setContentsMargins(0, 5, 0, 5)
        
        self.value_lbl = QLabel("--")
        self.value_lbl.setStyleSheet(f"color: {DarkTheme.TEXT_HIGHLIGHT}; font-size: 24px; font-weight: bold; font-family: 'Segoe UI', sans-serif;")
        val_layout.addWidget(self.value_lbl)
        
        if suffix:
            self.suffix_lbl = QLabel(suffix)
            self.suffix_lbl.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 14px; margin-top: 8px; font-weight: 500;")
            val_layout.addWidget(self.suffix_lbl)
            
        val_layout.addStretch()
        layout.addLayout(val_layout)
        
        # Status/Subtext
        self.status_lbl = QLabel("Waiting for data...")
        self.status_lbl.setStyleSheet(f"color: {DarkTheme.TEXT_SECONDARY}; font-size: 11px; font-style: italic;")
        layout.addWidget(self.status_lbl)

    def set_value(self, value_text, color=None, subtext=None):
        self.value_lbl.setText(value_text)
        if color:
            self.value_lbl.setStyleSheet(f"color: {color}; font-size: 24px; font-weight: bold; font-family: 'Segoe UI', sans-serif;")
        
        if subtext:
            self.status_lbl.setText(subtext)
            # If color is provided, tint the subtext slightly or keep it grey but specific
            self.status_lbl.setStyleSheet(f"color: {color if color else DarkTheme.TEXT_SECONDARY}; font-size: 11px;")
        else:
            self.status_lbl.setText("")


class Tab11RiskMetrics(QWidget):
    """Advanced Risk Metrics Dashboard - Tab 11 (Improved)"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.backtester = parent.backtester if hasattr(parent, "backtester") else None
        self.analysis_engines = parent.analysis_engines if hasattr(parent, "analysis_engines") else None

        # Initialize data storage
        self.current_results = None
        self.mae_mfe_data = None
        self.stress_test_results = None

        self.init_ui()
        self.setup_connections()
        self.load_initial_data()

    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(f"background-color: {DarkTheme.BG_SECONDARY}; border-bottom: 1px solid {DarkTheme.BORDER_COLOR};")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        header_layout.addStretch()

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.setStyleSheet(self.get_button_style(DarkTheme.INFO))
        self.refresh_btn.clicked.connect(self.refresh_data)
        header_layout.addWidget(self.refresh_btn)

        layout.addWidget(header)

        # Create splitter for main content
        splitter = QSplitter(Qt.Vertical)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet(
            f"""
            QSplitter::handle {{
                background-color: {DarkTheme.BG_HOVER};
            }}
        """
        )

        # Top section - Key Risk Metrics
        self.create_risk_metrics_section()
        splitter.addWidget(self.risk_metrics_group)

        # Middle section - Charts and Visualizations
        self.create_visualization_section()
        splitter.addWidget(self.visualization_group)

        # Bottom section - Detailed Analysis
        self.create_detailed_analysis_section()
        splitter.addWidget(self.detailed_analysis_group)

        # Set splitter proportions
        splitter.setStretchFactor(0, 20)
        splitter.setStretchFactor(1, 55)
        splitter.setStretchFactor(2, 25)
        splitter.setSizes([200, 550, 250])
        splitter.setCollapsible(0, False)

        layout.addWidget(splitter)

        # Status bar
        self.status_label = QLabel("Ready - Load backtest results to view risk metrics")
        self.status_label.setStyleSheet("color: #888888; font-style: italic; padding: 5px 10px; background-color: #1e1e1e;")
        layout.addWidget(self.status_label)

    def create_risk_metrics_section(self):
        """Create the key risk metrics display section"""
        self.risk_metrics_group = QGroupBox("🎯 Key Risk Metrics")
        self.risk_metrics_group.setStyleSheet(self.get_group_style())
        
        # Use 4 columns for better density
        layout = QGridLayout()
        layout.setContentsMargins(15, 25, 15, 15)
        layout.setSpacing(12)

        # Define all metrics in one list
        # (Label, Key, Suffix, Tooltip)
        all_metrics = [
            ("Sharpe Ratio", "sharpe", "", "Risk-adjusted return (higher is better)"),
            ("Sortino Ratio", "sortino", "", "Downside risk-adjusted return"),
            ("Calmar Ratio", "calmar", "", "Annual Return / Max Drawdown"),
            ("Recovery Factor", "recovery_factor", "", "Net Profit / Max Drawdown"),
            
            ("Max Drawdown", "max_dd", "%", "Maximum peak-to-valley decline"),
            ("Value at Risk (95%)", "var_95", "%", "Max expected loss with 95% confidence"),
            ("Exp. Shortfall (95%)", "es_95", "%", "Average loss in worst 5% cases"),
            ("Win Rate", "win_rate", "%", "Percentage of profitable trades"),

            ("Avg MAE", "avg_mae", "%", "Average Maximum Adverse Excursion (Pain)"),
            ("Avg MFE", "avg_mfe", "%", "Average Maximum Favorable Excursion (Potential)"),
            ("MAE/MFE Ratio", "mae_mfe_ratio", "", "Efficiency of entry vs exit"),
            ("Profit Factor", "profit_factor", "", "Gross Profit / Gross Loss"),
        ]

        self.metric_cards = {}
        
        for i, (label, key, suffix, tooltip) in enumerate(all_metrics):
            row = i // 4
            col = i % 4
            
            card = MetricCard(label, suffix, tooltip)
            layout.addWidget(card, row, col)
            self.metric_cards[key] = card

        self.risk_metrics_group.setLayout(layout)

    def create_visualization_section(self):
        """Create the visualization section with charts"""
        self.visualization_group = QGroupBox("📈 Risk Visualizations")
        self.visualization_group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 25, 15, 15)

        # Chart selector
        chart_controls = QHBoxLayout()
        
        chart_label = QLabel("Chart Type:")
        chart_label.setStyleSheet("color: #cccccc;")
        chart_controls.addWidget(chart_label)

        self.chart_combo = QComboBox()
        self.chart_combo.addItems(
            [
                "MAE/MFE Distribution",
                "Drawdown Analysis",
                "Volatility Clustering",
                "Stress Test Scenarios",
                "Risk-Return Scatter",
                "Tail Risk Analysis",
            ]
        )
        self.chart_combo.currentTextChanged.connect(self.update_chart)
        self.chart_combo.setStyleSheet(self.get_combo_style())
        chart_controls.addWidget(self.chart_combo)

        chart_controls.addStretch()
        
        self.log_scale_cb = QPushButton("Log Scale")
        self.log_scale_cb.setCheckable(True)
        self.log_scale_cb.clicked.connect(self.update_chart)
        self.log_scale_cb.setStyleSheet(
            """
            QPushButton {
                background-color: #3e3e3e;
                color: #ffffff;
                border: none;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #0e639c;
            }
            """
        )
        chart_controls.addWidget(self.log_scale_cb)
        
        layout.addLayout(chart_controls)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 6), facecolor='#1e1e1e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setStyleSheet("background-color: #1e1e1e;")
        layout.addWidget(self.canvas)

        self.visualization_group.setLayout(layout)

    def create_detailed_analysis_section(self):
        """Create detailed analysis section"""
        self.detailed_analysis_group = QGroupBox("📋 Detailed Analysis")
        self.detailed_analysis_group.setStyleSheet(self.get_group_style())
        
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 25, 15, 15)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setStyleSheet(
            """
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 10px;
                font-family: 'Consolas', monospace;
            }
        """
        )
        layout.addWidget(self.analysis_text)
        
        self.detailed_analysis_group.setLayout(layout)

    def get_group_style(self):
        return """
            QGroupBox {
                background-color: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 8px;
                margin-top: 12px;
                font-weight: bold;
                color: #ffffff;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 5px;
                background-color: #2d2d2d;
            }
        """

    def get_button_style(self, color):
        return f"""
            QPushButton {{
                background: {color};
                color: #1e1e1e;
                border: none;
                padding: 8px 16px;
                font-weight: bold;
                border-radius: 4px;
            }}
            QPushButton:hover {{ background: #ffffff; }}
        """

    def get_combo_style(self):
        return """
            QComboBox {
                background-color: #1e1e1e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 5px;
                color: #ffffff;
                min-width: 200px;
            }
            QComboBox::drop-down {
                border: none;
            }
        """

    def setup_connections(self):
        """Setup signal connections"""
        pass

    def load_initial_data(self):
        """Load initial data if available"""
        QTimer.singleShot(1000, self.refresh_data)

    def refresh_data(self):
        """Refresh data from parent platform"""
        if not hasattr(self.parent, "last_backtest_results") or not self.parent.last_backtest_results:
            self.status_label.setText("No backtest results available. Please run a backtest first.")
            return

        try:
            results = self.parent.last_backtest_results
            self.current_results = results
            
            # Extract data
            equity_curve = results.get("equity_curve", [])
            trades = results.get("trades", [])
            
            if not equity_curve:
                self.status_label.setText("No equity curve data available.")
                return
                
            # Convert to Series/Array
            equity_series = pd.Series(equity_curve)
            returns_series = equity_series.pct_change().dropna()
            
            # Initialize Calculator
            calculator = RiskMetricsCalculator(returns_series)
            
            # 1. Calculate Core Metrics
            var_95 = calculator.calculate_var(0.95)
            cvar_95 = calculator.calculate_cvar(0.95)
            max_dd = calculator.calculate_max_drawdown(equity_series)
            sharpe = calculator.calculate_sharpe_ratio()
            sortino = calculator.calculate_sortino_ratio()
            
            # Update Labels
            self.update_metric_label("max_dd", max_dd * 100, f"{max_dd:.1%}")
            self.update_metric_label("var_95", var_95 * 100, f"{var_95:.2%}")
            self.update_metric_label("es_95", cvar_95 * 100, f"{cvar_95:.2%}")
            self.update_metric_label("sharpe", sharpe, f"{sharpe:.2f}")
            self.update_metric_label("sortino", sortino, f"{sortino:.2f}")
            
            # 2. Calculate MAE/MFE Metrics if trades available
            if trades:
                df_trades = pd.DataFrame(trades)
                
                # Basic Trade Metrics
                win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) if len(df_trades) > 0 else 0
                gross_profit = df_trades[df_trades['pnl'] > 0]['pnl'].sum()
                gross_loss = abs(df_trades[df_trades['pnl'] < 0]['pnl'].sum())
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                
                net_profit = df_trades['pnl'].sum()
                # Recovery Factor (Net Profit / Max Drawdown $)
                # Need Max DD in $ terms, not %. 
                # Approx: Net Profit / (Max DD % * Initial Capital)
                # For now, let's use a simplified version or just skip if we don't have capital
                recovery_factor = 0
                if max_dd > 0:
                     # Assuming starting capital is implicit in equity curve
                     # This is an approximation
                     recovery_factor = (equity_series.iloc[-1] / equity_series.iloc[0] - 1) / max_dd
                
                self.update_metric_label("win_rate", win_rate * 100, f"{win_rate:.1%}")
                self.update_metric_label("profit_factor", profit_factor, f"{profit_factor:.2f}")
                self.update_metric_label("recovery_factor", recovery_factor, f"{recovery_factor:.2f}")
                self.update_metric_label("calmar", recovery_factor, f"{recovery_factor:.2f}") # Using Recovery as proxy for Calmar for now

                if "mae" in df_trades.columns and "mfe" in df_trades.columns:
                    avg_mae = df_trades["mae"].mean()
                    avg_mfe = df_trades["mfe"].mean()
                    max_mae = df_trades["mae"].max()
                    max_mfe = df_trades["mfe"].max()
                    ratio = avg_mfe / avg_mae if avg_mae > 0 else 0
                    
                    self.update_metric_label("avg_mae", avg_mae * 100, f"{avg_mae:.2%}")
                    self.update_metric_label("avg_mfe", avg_mfe * 100, f"{avg_mfe:.2%}")
                    self.update_metric_label("mae_mfe_ratio", ratio, f"{ratio:.2f}")
                    self.update_metric_label("max_mae", max_mae * 100, f"{max_mae:.2%}")
                    self.update_metric_label("max_mfe", max_mfe * 100, f"{max_mfe:.2%}")
            
            # 3. Run Monte Carlo
            self.stress_test_results = calculator.monte_carlo_simulation(num_simulations=500, horizon=60)
            
            # Update Chart
            self.update_chart()
            
            # Update Analysis Text
            self.update_analysis_text(var_95, cvar_95, max_dd, sharpe)
            
            self.status_label.setText(f"Risk metrics updated at {pd.Timestamp.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            self.status_label.setText(f"Error calculating risk metrics: {str(e)}")
            print(f"Error in refresh_data: {e}")

    def update_metric_label(self, key, value, text):
        """Helper to update metric labels safely"""
        if key in self.metric_cards:
            card = self.metric_cards[key]
            
            # Color coding
            color = DarkTheme.TEXT_HIGHLIGHT
            subtext = ""
            
            if key in ["max_dd", "var_95", "es_95", "avg_mae", "max_mae"]:
                # Bad metrics (Red if high)
                if value > 20: # Critical
                    color = DarkTheme.ERROR
                    subtext = "Critical Level"
                elif value > 10: # Warning
                    color = "#f48771" # Light Red
                    subtext = "High Risk"
                else:
                    color = DarkTheme.SUCCESS
                    subtext = "Within Limits"
            elif key in ["sharpe", "sortino", "calmar", "profit_factor", "recovery_factor"]:
                # Good metrics (Green if high)
                if value > 2.0:
                    color = DarkTheme.SUCCESS
                    subtext = "Excellent"
                elif value > 1.0:
                    color = "#4ec9b0" # Light Green
                    subtext = "Good"
                else:
                    color = DarkTheme.ERROR
                    subtext = "Poor"
            
            card.set_value(text, color, subtext)

    def update_analysis_text(self, var, cvar, max_dd, sharpe):
        """Generate text analysis"""
        analysis = f"""
        RISK ANALYSIS REPORT
        ====================
        
        1. VALUE AT RISK (VaR 95%)
           - Daily VaR: {var:.2%}
           - Interpretation: In 95% of days, loss will not exceed {var:.2%}.
           - Conversely, expect a loss > {var:.2%} once every 20 days.
           
        2. EXPECTED SHORTFALL (CVaR 95%)
           - CVaR: {cvar:.2%}
           - Interpretation: When a tail event occurs (worst 5% days), 
             the average loss is expected to be {cvar:.2%}.
             
        3. DRAWDOWN ANALYSIS
           - Max Drawdown: {max_dd:.2%}
           - Recovery Factor: {sharpe * 2:.2f} (Estimated)
           
        4. STRESS TEST (Monte Carlo)
           - Simulations: 500
           - Horizon: 60 days
           - See 'Stress Test Scenarios' chart for projected paths.
        """
        self.analysis_text.setText(analysis)

    def update_chart(self):
        """Update chart based on selection"""
        chart_type = self.chart_combo.currentText()
        
        # Clear figure
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.set_facecolor('#1e1e1e')
        ax.tick_params(colors='#cccccc')
        ax.xaxis.label.set_color('#cccccc')
        ax.yaxis.label.set_color('#cccccc')
        ax.title.set_color('#ffffff')
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_color('#3e3e3e')
            
        if not self.current_results:
            ax.text(0.5, 0.5, "No Data Available", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='#666666', fontsize=14)
            self.canvas.draw()
            return

        try:
            if chart_type == "Stress Test Scenarios" and self.stress_test_results:
                paths = self.stress_test_results["paths"]
                # Plot first 50 paths
                for i in range(min(50, len(paths))):
                    ax.plot(paths[i], color='#4ec9b0', alpha=0.1)
                
                # Plot mean
                mean_path = np.mean(paths, axis=0)
                ax.plot(mean_path, color='#ffffff', linewidth=2, label='Mean')
                
                ax.set_title("Monte Carlo Simulation (Projected Equity)")
                ax.set_xlabel("Days Forward")
                ax.set_ylabel("Equity Multiplier")
                
            elif chart_type == "MAE/MFE Distribution":
                trades = self.current_results.get("trades", [])
                if trades:
                    df = pd.DataFrame(trades)
                    if "mae" in df.columns:
                        sns.histplot(df["mae"] * 100, ax=ax, color="#f48771", alpha=0.6, label="MAE", kde=True)
                    if "mfe" in df.columns:
                        sns.histplot(df["mfe"] * 100, ax=ax, color="#4ec9b0", alpha=0.6, label="MFE", kde=True)
                    ax.legend()
                    ax.set_title("MAE vs MFE Distribution")
                    ax.set_xlabel("Excursion (%)")
                    
            elif chart_type == "Drawdown Analysis":
                equity = pd.Series(self.current_results.get("equity_curve", []))
                peaks = equity.cummax()
                drawdown = (equity - peaks) / peaks * 100
                
                ax.fill_between(range(len(drawdown)), drawdown, 0, color="#f48771", alpha=0.3)
                ax.plot(drawdown, color="#f48771", linewidth=1)
                ax.set_title("Underwater Plot (Drawdown)")
                ax.set_ylabel("Drawdown (%)")
                
            else:
                ax.text(0.5, 0.5, f"{chart_type} not implemented yet", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes, color='#666666')
                        
        except Exception as e:
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='#f48771')
            print(f"Chart error: {e}")
            
        self.canvas.draw()
