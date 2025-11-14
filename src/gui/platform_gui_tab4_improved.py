from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView, QCheckBox, QPushButton, QFrame, QSplitter,
    QComboBox, QListWidget, QListWidgetItem, QMessageBox, QFileDialog, QTabWidget
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtWebEngineWidgets import QWebEngineView
import logging
import pandas as pd
import json
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class BacktestHistoryItem:
    """Class to represent a historical backtest result"""
    def __init__(self, name, timestamp, results, filepath=None):
        self.name = name
        self.timestamp = timestamp
        self.results = results
        self.filepath = filepath
        
    def get_metrics_summary(self):
        """Get key metrics for display"""
        metrics = self.results.get('metrics', {})
        return {
            'sharpe': metrics.get('sharpe', 0),
            'total_return': metrics.get('total_return', 0),
            'win_rate': metrics.get('win_rate', 0),
            'num_trades': metrics.get('num_trades', 0)
        }


class Tab4ResultsAnalysis(QWidget):
    """
    Enhanced Results Analysis Tab with:
    - Historical backtest comparison
    - Multiple visualization types
    - Advanced export options
    - Performance analytics
    """
    trade_clicked = Signal(dict)
    status_update = Signal(str, str)  # message, type

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.logger = logging.getLogger(__name__)
        self.backtest_history = []  # List of BacktestHistoryItem
        self.current_results = None
        self.selected_comparison = []  # For multi-backtest comparison

        self.init_ui()
        
        # Load historical backtests on init
        QTimer.singleShot(500, self.load_backtest_history)

    def init_ui(self):
        """Initialize the user interface"""
        main_layout = QHBoxLayout()
        
        # Left panel: Backtest history and comparison
        left_panel = self.create_left_panel()
        
        # Right panel: Charts and analysis
        right_panel = self.create_right_panel()
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([250, 750])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)

    def create_left_panel(self):
        """Create left panel with backtest history"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Title
        title = QLabel("📊 Backtest History")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        layout.addWidget(title)
        
        # Current result section
        current_group = QFrame()
        current_group.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-left: 4px solid #4ec9b0;
                border-radius: 6px;
                padding: 12px;
            }
        """)
        current_layout = QVBoxLayout()
        
        current_label = QLabel("<b>Current Result</b>")
        current_label.setStyleSheet("color: #4ec9b0;")
        current_layout.addWidget(current_label)
        
        self.current_result_text = QLabel("No backtest loaded")
        self.current_result_text.setWordWrap(True)
        self.current_result_text.setStyleSheet("color: #cccccc; font-size: 11px;")
        current_layout.addWidget(self.current_result_text)
        
        current_group.setLayout(current_layout)
        layout.addWidget(current_group)
        
        # History list
        history_label = QLabel("<b>Saved Backtests</b>")
        history_label.setStyleSheet("color: #cccccc; margin-top: 12px;")
        layout.addWidget(history_label)
        
        self.history_list = QListWidget()
        self.history_list.setStyleSheet("""
            QListWidget {
                background-color: #252525;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #3e3e3e;
            }
            QListWidget::item:selected {
                background-color: #0e639c;
            }
            QListWidget::item:hover {
                background-color: #2d2d2d;
            }
        """)
        self.history_list.itemClicked.connect(self.on_history_item_clicked)
        layout.addWidget(self.history_list, 1)
        
        # Action buttons
        actions_layout = QVBoxLayout()
        
        self.compare_btn = QPushButton("📊 Compare Selected")
        self.compare_btn.clicked.connect(self.compare_backtests)
        self.compare_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        self.compare_btn.setEnabled(False)
        actions_layout.addWidget(self.compare_btn)
        
        self.save_current_btn = QPushButton("💾 Save Current")
        self.save_current_btn.clicked.connect(self.save_current_backtest)
        self.save_current_btn.setStyleSheet(self.compare_btn.styleSheet())
        self.save_current_btn.setEnabled(False)
        actions_layout.addWidget(self.save_current_btn)
        
        self.delete_btn = QPushButton("🗑️ Delete Selected")
        self.delete_btn.clicked.connect(self.delete_selected_backtest)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: #f48771;
                color: #1e1e1e;
                font-weight: bold;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #ff9f8f;
            }
        """)
        actions_layout.addWidget(self.delete_btn)
        
        layout.addLayout(actions_layout)
        
        # Multi-select checkbox
        self.multiselect_check = QCheckBox("Multi-select mode")
        self.multiselect_check.stateChanged.connect(self.toggle_multiselect)
        self.multiselect_check.setStyleSheet("color: #cccccc; margin-top: 8px;")
        layout.addWidget(self.multiselect_check)
        
        widget.setLayout(layout)
        return widget
        
    def create_right_panel(self):
        """Create right panel with charts and tables"""
        widget = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(12)
        
        # Header with export options
        header_layout = QHBoxLayout()
        
        title = QLabel("📈 Results Analysis")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #ffffff;")
        header_layout.addWidget(title)
        header_layout.addStretch()
        
        # Chart type selector
        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Equity Curve",
            "Drawdown Analysis",
            "Win/Loss Distribution",
            "Monthly Returns",
            "Trade Timeline",
            "Risk Metrics"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.on_chart_type_changed)
        self.chart_type_combo.setMinimumWidth(180)
        header_layout.addWidget(QLabel("Chart:"))
        header_layout.addWidget(self.chart_type_combo)
        
        # Export buttons
        self.export_pdf_btn = QPushButton("📄 PDF Report")
        self.export_pdf_btn.clicked.connect(self.export_pdf_report)
        self.export_pdf_btn.setStyleSheet("""
            QPushButton {
                background-color: #0e639c;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #1177bb;
            }
        """)
        header_layout.addWidget(self.export_pdf_btn)
        
        self.export_csv_btn = QPushButton("📋 CSV")
        self.export_csv_btn.clicked.connect(self.export_csv)
        self.export_csv_btn.setStyleSheet(self.export_pdf_btn.styleSheet())
        header_layout.addWidget(self.export_csv_btn)
        
        layout.addLayout(header_layout)
        
        # Main chart area
        self.chart_view = QWebEngineView()
        self.chart_view.setMinimumHeight(400)
        layout.addWidget(self.chart_view, 2)
        
        # Metrics cards
        metrics_layout = self.create_metrics_cards()
        layout.addLayout(metrics_layout)
        
        # Bottom section with tables
        bottom_tabs = QTabWidget()
        
        # Trades tab
        trades_tab = self.create_trades_tab()
        bottom_tabs.addTab(trades_tab, "Trades Log")
        
        # Statistics tab
        stats_tab = self.create_statistics_tab()
        bottom_tabs.addTab(stats_tab, "Statistics")
        
        # Recommendations tab
        recommendations_tab = self.create_recommendations_tab()
        bottom_tabs.addTab(recommendations_tab, "Recommendations")
        
        layout.addWidget(bottom_tabs, 1)
        
        widget.setLayout(layout)
        return widget
        
    def create_metrics_cards(self):
        """Create metric display cards"""
        layout = QHBoxLayout()
        
        self.metric_sharpe = self.create_metric_card("Sharpe Ratio", "0.00", "#4ec9b0")
        self.metric_return = self.create_metric_card("Total Return", "0.00%", "#569cd6")
        self.metric_winrate = self.create_metric_card("Win Rate", "0.00%", "#dcdcaa")
        self.metric_trades = self.create_metric_card("Total Trades", "0", "#c586c0")
        self.metric_maxdd = self.create_metric_card("Max Drawdown", "0.00%", "#f48771")
        
        layout.addWidget(self.metric_sharpe)
        layout.addWidget(self.metric_return)
        layout.addWidget(self.metric_winrate)
        layout.addWidget(self.metric_trades)
        layout.addWidget(self.metric_maxdd)
        
        return layout
        
    def create_metric_card(self, title, value, color):
        """Create a metric display card"""
        frame = QFrame()
        frame.setStyleSheet(f"""
            QFrame {{
                background-color: #2d2d2d;
                border-left: 3px solid {color};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        
        layout = QVBoxLayout()
        layout.setSpacing(4)
        
        title_label = QLabel(title)
        title_label.setStyleSheet("color: #cccccc; font-size: 10px;")
        
        value_label = QLabel(value)
        value_label.setObjectName("valueLabel")
        value_label.setStyleSheet(f"color: {color}; font-size: 18px; font-weight: bold;")
        
        layout.addWidget(title_label)
        layout.addWidget(value_label)
        
        frame.setLayout(layout)
        return frame
        
    def create_trades_tab(self):
        """Create trades log tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Filter controls
        filter_layout = QHBoxLayout()
        
        self.score_filter = QCheckBox("High score only (≥4)")
        self.score_filter.stateChanged.connect(self.apply_trade_filters)
        filter_layout.addWidget(self.score_filter)
        
        self.winners_only = QCheckBox("Winners only")
        self.winners_only.stateChanged.connect(self.apply_trade_filters)
        filter_layout.addWidget(self.winners_only)
        
        filter_layout.addStretch()
        
        export_trades_btn = QPushButton("💾 Export Trades")
        export_trades_btn.clicked.connect(self.export_trades)
        filter_layout.addWidget(export_trades_btn)
        
        layout.addLayout(filter_layout)
        
        # Trades table
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(10)
        self.trade_table.setHorizontalHeaderLabels([
            "Timestamp", "Direction", "Entry", "Exit", "PnL%", 
            "Score", "Duration", "Type", "Exit Reason", "MAE%"
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.itemDoubleClicked.connect(self.on_trade_double_clicked)
        layout.addWidget(self.trade_table)
        
        widget.setLayout(layout)
        return widget
        
    def create_statistics_tab(self):
        """Create statistics comparison tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Split by entry quality
        split_layout = QHBoxLayout()
        
        # Good entries
        good_group = QGroupBox("✓ High Quality Entries (Score ≥ 4)")
        good_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #4ec9b0;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        good_layout = QVBoxLayout()
        
        self.good_stats_table = QTableWidget()
        self.good_stats_table.setColumnCount(2)
        self.good_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.good_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        good_layout.addWidget(self.good_stats_table)
        
        good_group.setLayout(good_layout)
        split_layout.addWidget(good_group)
        
        # Bad entries
        bad_group = QGroupBox("⚠️ Low Quality Entries (Score < 4)")
        bad_group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 2px solid #f48771;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
            }
        """)
        bad_layout = QVBoxLayout()
        
        self.bad_stats_table = QTableWidget()
        self.bad_stats_table.setColumnCount(2)
        self.bad_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.bad_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        bad_layout.addWidget(self.bad_stats_table)
        
        bad_group.setLayout(bad_layout)
        split_layout.addWidget(bad_group)
        
        layout.addLayout(split_layout)
        
        widget.setLayout(layout)
        return widget
        
    def create_recommendations_tab(self):
        """Create recommendations tab"""
        widget = QWidget()
        layout = QVBoxLayout()
        
        # AI-style recommendations
        rec_frame = QFrame()
        rec_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-left: 4px solid #569cd6;
                border-radius: 6px;
                padding: 16px;
            }
        """)
        rec_layout = QVBoxLayout()
        
        rec_title = QLabel("💡 <b>Performance Insights</b>")
        rec_title.setStyleSheet("color: #569cd6; font-size: 14px;")
        rec_layout.addWidget(rec_title)
        
        self.recommendation_text = QLabel("Run a backtest to see recommendations")
        self.recommendation_text.setWordWrap(True)
        self.recommendation_text.setStyleSheet("color: #cccccc; margin-top: 8px;")
        rec_layout.addWidget(self.recommendation_text)
        
        rec_frame.setLayout(rec_layout)
        layout.addWidget(rec_frame)
        
        # Action items
        actions_frame = QFrame()
        actions_frame.setStyleSheet("""
            QFrame {
                background-color: #2d2d2d;
                border-left: 4px solid #dcdcaa;
                border-radius: 6px;
                padding: 16px;
                margin-top: 12px;
            }
        """)
        actions_layout = QVBoxLayout()
        
        actions_title = QLabel("🎯 <b>Suggested Actions</b>")
        actions_title.setStyleSheet("color: #dcdcaa; font-size: 14px;")
        actions_layout.addWidget(actions_title)
        
        self.actions_text = QLabel("Actions will appear here based on results")
        self.actions_text.setWordWrap(True)
        self.actions_text.setStyleSheet("color: #cccccc; margin-top: 8px;")
        actions_layout.addWidget(self.actions_text)
        
        actions_frame.setLayout(actions_layout)
        layout.addWidget(actions_frame)
        
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
        
    def on_tab_activated(self):
        """Called when this tab becomes active"""
        if hasattr(self.parent_platform, 'last_backtest_results') and self.parent_platform.last_backtest_results:
            self.load_results(self.parent_platform.last_backtest_results)
            self.save_current_btn.setEnabled(True)
        else:
            self.clear_display()
            
    def load_backtest_history(self):
        """Load saved backtest results from disk"""
        try:
            results_dir = "results/backtests"
            if not os.path.exists(results_dir):
                os.makedirs(results_dir, exist_ok=True)
                return
                
            self.backtest_history.clear()
            self.history_list.clear()
            
            for filename in os.listdir(results_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(results_dir, filename)
                    try:
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            
                        name = data.get('name', filename.replace('.json', ''))
                        timestamp = data.get('timestamp', 'Unknown')
                        results = data.get('results', {})
                        
                        item = BacktestHistoryItem(name, timestamp, results, filepath)
                        self.backtest_history.append(item)
                        
                        # Add to list widget
                        summary = item.get_metrics_summary()
                        list_item = QListWidgetItem(
                            f"{name}\n"
                            f"Sharpe: {summary['sharpe']:.2f} | "
                            f"Return: {summary['total_return']*100:.1f}% | "
                            f"Trades: {summary['num_trades']}"
                        )
                        self.history_list.addItem(list_item)
                        
                    except Exception as e:
                        self.logger.error(f"Error loading {filename}: {e}")
                        
            self.status_update.emit(f"Loaded {len(self.backtest_history)} historical backtests", "info")
            
        except Exception as e:
            self.logger.error(f"Error loading backtest history: {e}")
            
    def load_results(self, results):
        """Load and display backtest results"""
        try:
            self.current_results = results
            
            # Update metrics cards
            self.update_metrics(results)
            
            # Load trade data
            self.load_trade_data(results)
            
            # Calculate statistics
            self.calculate_statistics(results)
            
            # Generate recommendations
            self.generate_recommendations(results)
            
            # Render current chart
            self.render_current_chart(results)
            
            # Update current result display
            metrics = results.get('metrics', {})
            self.current_result_text.setText(
                f"Sharpe: {metrics.get('sharpe', 0):.2f}\n"
                f"Return: {metrics.get('total_return', 0)*100:.1f}%\n"
                f"Trades: {metrics.get('num_trades', 0)}"
            )
            
            self.status_update.emit("Results loaded successfully", "success")
            
        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load results:\n{str(e)}")
            
    def update_metrics(self, results):
        """Update metric display cards"""
        metrics = results.get('metrics', {})
        
        sharpe = metrics.get('sharpe', 0)
        total_return = metrics.get('total_return', 0) * 100
        win_rate = metrics.get('win_rate', 0) * 100
        num_trades = metrics.get('num_trades', 0)
        max_dd = abs(metrics.get('max_dd', 0)) * 100
        
        self.metric_sharpe.findChild(QLabel, "valueLabel").setText(f"{sharpe:.2f}")
        self.metric_return.findChild(QLabel, "valueLabel").setText(f"{total_return:.1f}%")
        self.metric_winrate.findChild(QLabel, "valueLabel").setText(f"{win_rate:.1f}%")
        self.metric_trades.findChild(QLabel, "valueLabel").setText(str(num_trades))
        self.metric_maxdd.findChild(QLabel, "valueLabel").setText(f"{max_dd:.1f}%")
        
    def render_current_chart(self, results):
        """Render currently selected chart type"""
        chart_type = self.chart_type_combo.currentText()
        
        if chart_type == "Equity Curve":
            self.render_equity_chart(results)
        elif chart_type == "Drawdown Analysis":
            self.render_drawdown_chart(results)
        elif chart_type == "Win/Loss Distribution":
            self.render_distribution_chart(results)
        elif chart_type == "Monthly Returns":
            self.render_monthly_returns(results)
        elif chart_type == "Trade Timeline":
            self.render_trade_timeline(results)
        elif chart_type == "Risk Metrics":
            self.render_risk_metrics(results)
            
    def on_chart_type_changed(self, chart_type):
        """Handle chart type selection change"""
        if self.current_results:
            self.render_current_chart(self.current_results)
            
    def render_equity_chart(self, results):
        """Render equity curve with comparison if multiple selected"""
        try:
            fig = go.Figure()
            
            # Current results
            if 'equity_curve' in results:
                equity = results['equity_curve']
                fig.add_trace(go.Scatter(
                    y=equity,
                    mode='lines',
                    name='Current',
                    line=dict(color='#4ec9b0', width=2)
                ))
                
            # Add comparison backtests if any
            for item in self.selected_comparison:
                if 'equity_curve' in item.results:
                    fig.add_trace(go.Scatter(
                        y=item.results['equity_curve'],
                        mode='lines',
                        name=item.name,
                        line=dict(width=1.5),
                        opacity=0.7
                    ))
                    
            fig.update_layout(
                title="Equity Curve Comparison",
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                template='plotly_dark',
                height=400,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc'),
                hovermode='x unified'
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error rendering equity chart: {e}")
            self.chart_view.setHtml(f"<h3 style='color:#f48771'>Error: {str(e)}</h3>")
            
    def render_drawdown_chart(self, results):
        """Render drawdown analysis"""
        try:
            if 'equity_curve' not in results:
                self.chart_view.setHtml("<h3>No equity data available</h3>")
                return
                
            equity = pd.Series(results['equity_curve'])
            cummax = equity.cummax()
            drawdown = (equity - cummax) / cummax * 100
            
            fig = make_subplots(rows=2, cols=1, row_heights=[0.6, 0.4],
                               subplot_titles=('Equity Curve', 'Drawdown %'))
            
            # Equity
            fig.add_trace(
                go.Scatter(y=equity, mode='lines', name='Equity',
                          line=dict(color='#4ec9b0', width=2)),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(y=drawdown, mode='lines', name='Drawdown',
                          fill='tozeroy', line=dict(color='#f48771', width=1)),
                row=2, col=1
            )
            
            fig.update_layout(
                template='plotly_dark',
                height=400,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc'),
                showlegend=False
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error rendering drawdown chart: {e}")
            
    def render_distribution_chart(self, results):
        """Render win/loss distribution histogram"""
        try:
            if 'trades' not in results or not results['trades']:
                self.chart_view.setHtml("<h3>No trade data available</h3>")
                return
                
            trades_df = pd.DataFrame(results['trades'])
            if 'pnl_pct' not in trades_df.columns:
                return
                
            fig = go.Figure()
            
            # Winners
            wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct']
            if not wins.empty:
                fig.add_trace(go.Histogram(
                    x=wins, name="Wins",
                    marker_color='#4ec9b0',
                    opacity=0.75, nbinsx=30
                ))
                
            # Losers
            losses = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct']
            if not losses.empty:
                fig.add_trace(go.Histogram(
                    x=losses, name="Losses",
                    marker_color='#f48771',
                    opacity=0.75, nbinsx=30
                ))
                
            fig.update_layout(
                title="Win/Loss Distribution",
                xaxis_title="PnL %",
                yaxis_title="Frequency",
                barmode='overlay',
                template='plotly_dark',
                height=400,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc')
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error rendering distribution chart: {e}")
            
    def render_monthly_returns(self, results):
        """Render monthly returns heatmap"""
        try:
            # Placeholder - would need timestamp data to calculate monthly returns
            self.chart_view.setHtml(
                "<div style='color:#cccccc; padding:20px;'>"
                "<h3>Monthly Returns Heatmap</h3>"
                "<p>Feature requires timestamped equity curve data</p>"
                "</div>"
            )
        except Exception as e:
            self.logger.error(f"Error rendering monthly returns: {e}")
            
    def render_trade_timeline(self, results):
        """Render trades on timeline"""
        try:
            if 'trades' not in results or not results['trades']:
                self.chart_view.setHtml("<h3>No trade data available</h3>")
                return
                
            trades_df = pd.DataFrame(results['trades'])
            
            fig = go.Figure()
            
            # Plot trades as scatter
            colors = ['#4ec9b0' if pnl > 0 else '#f48771' for pnl in trades_df.get('pnl_pct', [])]
            sizes = [abs(pnl) * 2 + 5 for pnl in trades_df.get('pnl_pct', [])]
            
            fig.add_trace(go.Scatter(
                x=list(range(len(trades_df))),
                y=trades_df.get('pnl_pct', []),
                mode='markers',
                marker=dict(
                    size=sizes,
                    color=colors,
                    line=dict(width=1, color='#1e1e1e')
                ),
                text=[f"Trade {i+1}<br>PnL: {pnl:.2f}%" 
                      for i, pnl in enumerate(trades_df.get('pnl_pct', []))],
                hoverinfo='text'
            ))
            
            fig.update_layout(
                title="Trade Timeline",
                xaxis_title="Trade Number",
                yaxis_title="PnL %",
                template='plotly_dark',
                height=400,
                paper_bgcolor='#1e1e1e',
                plot_bgcolor='#1e1e1e',
                font=dict(color='#cccccc')
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error rendering trade timeline: {e}")
            
    def render_risk_metrics(self, results):
        """Render risk metrics dashboard"""
        try:
            metrics = results.get('metrics', {})
            
            # Create gauge charts for risk metrics
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=('Sharpe Ratio', 'Max Drawdown', 'Win Rate')
            )
            
            # Sharpe gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('sharpe', 0),
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [-1, 3]},
                       'bar': {'color': '#4ec9b0'},
                       'steps': [
                           {'range': [-1, 0], 'color': '#f48771'},
                           {'range': [0, 1], 'color': '#dcdcaa'},
                           {'range': [1, 3], 'color': '#4ec9b0'}
                       ]}
            ), row=1, col=1)
            
            # Max DD gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=abs(metrics.get('max_dd', 0)) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 50]},
                       'bar': {'color': '#f48771'}}
            ), row=1, col=2)
            
            # Win rate gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=metrics.get('win_rate', 0) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': '#569cd6'}}
            ), row=1, col=3)
            
            fig.update_layout(
                template='plotly_dark',
                height=300,
                paper_bgcolor='#1e1e1e',
                font=dict(color='#cccccc')
            )
            
            html = fig.to_html(include_plotlyjs='cdn')
            self.chart_view.setHtml(html)
            
        except Exception as e:
            self.logger.error(f"Error rendering risk metrics: {e}")
            
    def load_trade_data(self, results):
        """Load trades into table"""
        try:
            if 'trades' not in results or not results['trades']:
                self.trade_table.setRowCount(0)
                return
                
            trades = results['trades']
            self.trade_table.setRowCount(len(trades))
            
            for i, trade in enumerate(trades):
                # Extract trade data
                timestamp = trade.get('timestamp', str(i))
                direction = "LONG" if trade.get('pnl_pct', 0) > 0 else "SHORT"
                entry = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                score = trade.get('score', 0)
                duration = trade.get('duration', 'N/A')
                trade_type = trade.get('entry_type', 'N/A')
                exit_reason = trade.get('reason_exit', 'N/A')
                mae_pct = trade.get('mae_pct', 0)
                
                # Populate table
                self.trade_table.setItem(i, 0, QTableWidgetItem(str(timestamp)))
                
                dir_item = QTableWidgetItem(direction)
                dir_item.setForeground(Qt.GlobalColor.green if direction == "LONG" else Qt.GlobalColor.red)
                self.trade_table.setItem(i, 1, dir_item)
                
                self.trade_table.setItem(i, 2, QTableWidgetItem(f"{entry:.2f}"))
                self.trade_table.setItem(i, 3, QTableWidgetItem(f"{exit_price:.2f}"))
                
                pnl_item = QTableWidgetItem(f"{pnl_pct:.2f}%")
                pnl_item.setForeground(Qt.GlobalColor.green if pnl_pct > 0 else Qt.GlobalColor.red)
                self.trade_table.setItem(i, 4, pnl_item)
                
                score_item = QTableWidgetItem(str(score))
                score_item.setBackground(Qt.GlobalColor.green if score >= 4 else Qt.GlobalColor.red)
                self.trade_table.setItem(i, 5, score_item)
                
                self.trade_table.setItem(i, 6, QTableWidgetItem(str(duration)))
                self.trade_table.setItem(i, 7, QTableWidgetItem(trade_type))
                self.trade_table.setItem(i, 8, QTableWidgetItem(exit_reason))
                self.trade_table.setItem(i, 9, QTableWidgetItem(f"{mae_pct:.2f}%"))
                
        except Exception as e:
            self.logger.error(f"Error loading trade data: {e}")
            
    def apply_trade_filters(self):
        """Apply filters to trade table"""
        if not self.current_results:
            return
            
        # Reload with filters
        self.load_trade_data(self.current_results)
        
        # Hide rows that don't match filter
        for row in range(self.trade_table.rowCount()):
            show_row = True
            
            if self.score_filter.isChecked():
                score_item = self.trade_table.item(row, 5)
                if score_item and int(score_item.text()) < 4:
                    show_row = False
                    
            if self.winners_only.isChecked():
                pnl_item = self.trade_table.item(row, 4)
                if pnl_item:
                    pnl_value = float(pnl_item.text().replace('%', ''))
                    if pnl_value <= 0:
                        show_row = False
                        
            self.trade_table.setRowHidden(row, not show_row)
            
    def calculate_statistics(self, results):
        """Calculate and display statistics"""
        try:
            if 'trades' not in results or not results['trades']:
                return
                
            trades_df = pd.DataFrame(results['trades'])
            
            # Good trades (score >= 4)
            good_trades = trades_df[trades_df.get('score', 0) >= 4]
            if not good_trades.empty:
                good_stats = {
                    "Total Trades": len(good_trades),
                    "Win Rate": f"{(good_trades['pnl_pct'] > 0).mean() * 100:.1f}%",
                    "Avg PnL": f"{good_trades['pnl_pct'].mean():.2f}%",
                    "Total PnL": f"{good_trades['pnl_pct'].sum():.2f}%",
                    "Best Trade": f"{good_trades['pnl_pct'].max():.2f}%",
                    "Worst Trade": f"{good_trades['pnl_pct'].min():.2f}%"
                }
                self.populate_stats_table(self.good_stats_table, good_stats)
            else:
                self.good_stats_table.setRowCount(0)
                
            # Bad trades (score < 4)
            bad_trades = trades_df[trades_df.get('score', 0) < 4]
            if not bad_trades.empty:
                bad_stats = {
                    "Total Trades": len(bad_trades),
                    "Win Rate": f"{(bad_trades['pnl_pct'] > 0).mean() * 100:.1f}%",
                    "Avg PnL": f"{bad_trades['pnl_pct'].mean():.2f}%",
                    "Total PnL": f"{bad_trades['pnl_pct'].sum():.2f}%",
                    "Best Trade": f"{bad_trades['pnl_pct'].max():.2f}%",
                    "Worst Trade": f"{bad_trades['pnl_pct'].min():.2f}%"
                }
                self.populate_stats_table(self.bad_stats_table, bad_stats)
            else:
                self.bad_stats_table.setRowCount(0)
                
        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")
            
    def populate_stats_table(self, table, stats_dict):
        """Populate a statistics table"""
        table.setRowCount(len(stats_dict))
        for i, (metric, value) in enumerate(stats_dict.items()):
            table.setItem(i, 0, QTableWidgetItem(metric))
            table.setItem(i, 1, QTableWidgetItem(str(value)))
            
    def generate_recommendations(self, results):
        """Generate AI-style recommendations"""
        try:
            metrics = results.get('metrics', {})
            trades_df = pd.DataFrame(results.get('trades', []))
            
            sharpe = metrics.get('sharpe', 0)
            win_rate = metrics.get('win_rate', 0)
            max_dd = abs(metrics.get('max_dd', 0))
            
            # Generate insights
            insights = []
            actions = []
            
            if sharpe > 1.5:
                insights.append(f"✓ Excellent risk-adjusted returns (Sharpe: {sharpe:.2f})")
                actions.append("Consider increasing position size")
            elif sharpe < 0.5:
                insights.append(f"⚠️ Poor risk-adjusted returns (Sharpe: {sharpe:.2f})")
                actions.append("Review entry criteria and risk management")
                
            if win_rate > 0.6:
                insights.append(f"✓ High win rate ({win_rate*100:.1f}%)")
            elif win_rate < 0.4:
                insights.append(f"⚠️ Low win rate ({win_rate*100:.1f}%)")
                actions.append("Focus on improving entry quality")
                
            if max_dd > 0.25:
                insights.append(f"⚠️ High maximum drawdown ({max_dd*100:.1f}%)")
                actions.append("Implement tighter stop-losses")
            else:
                insights.append(f"✓ Acceptable drawdown control ({max_dd*100:.1f}%)")
                
            # Trade quality analysis
            if not trades_df.empty and 'score' in trades_df.columns:
                good_trades = trades_df[trades_df['score'] >= 4]
                if not good_trades.empty:
                    good_winrate = (good_trades['pnl_pct'] > 0).mean()
                    if good_winrate > 0.65:
                        insights.append(f"✓ High-quality entries performing well ({good_winrate*100:.1f}% win rate)")
                        actions.append("Focus on score≥4 entries only")
                        
            # Set recommendations
            self.recommendation_text.setText("\n".join(insights))
            self.actions_text.setText("\n".join(actions) if actions else "No critical actions needed")
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            
    def on_history_item_clicked(self, item):
        """Handle clicking on historical backtest"""
        index = self.history_list.row(item)
        if 0 <= index < len(self.backtest_history):
            historical_item = self.backtest_history[index]
            
            if self.multiselect_check.isChecked():
                # Add to comparison
                if historical_item not in self.selected_comparison:
                    self.selected_comparison.append(historical_item)
                    self.compare_btn.setEnabled(len(self.selected_comparison) > 0)
            else:
                # Load this result
                self.load_results(historical_item.results)
                
    def toggle_multiselect(self, state):
        """Toggle multi-select mode"""
        if state == Qt.CheckState.Checked.value:
            self.history_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            self.selected_comparison.clear()
        else:
            self.history_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
            self.selected_comparison.clear()
            self.compare_btn.setEnabled(False)
            
    def compare_backtests(self):
        """Compare multiple backtests"""
        if not self.selected_comparison:
            return
            
        # Render equity curves together
        if self.current_results:
            self.render_equity_chart(self.current_results)
            
    def save_current_backtest(self):
        """Save current backtest to history"""
        if not self.current_results:
            QMessageBox.warning(self, "Error", "No backtest results to save")
            return
            
        # Ask for name
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Backtest", "Enter backtest name:")
        
        if ok and name:
            try:
                os.makedirs("results/backtests", exist_ok=True)
                
                timestamp = datetime.now().isoformat()
                filepath = f"results/backtests/{name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                
                data = {
                    'name': name,
                    'timestamp': timestamp,
                    'results': self.current_results
                }
                
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
                QMessageBox.information(self, "Success", f"Backtest saved as:\n{name}")
                
                # Reload history
                self.load_backtest_history()
                
                self.status_update.emit(f"Saved backtest: {name}", "success")
                
            except Exception as e:
                self.logger.error(f"Error saving backtest: {e}")
                QMessageBox.critical(self, "Error", f"Failed to save:\n{str(e)}")
                
    def delete_selected_backtest(self):
        """Delete selected historical backtest"""
        selected_items = self.history_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "No backtest selected")
            return
            
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete {len(selected_items)} backtest(s)?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            for item in selected_items:
                index = self.history_list.row(item)
                historical_item = self.backtest_history[index]
                
                if historical_item.filepath and os.path.exists(historical_item.filepath):
                    os.remove(historical_item.filepath)
                    
            self.load_backtest_history()
            self.status_update.emit(f"Deleted {len(selected_items)} backtest(s)", "success")
            
    def export_pdf_report(self):
        """Export comprehensive PDF report"""
        QMessageBox.information(
            self,
            "PDF Export",
            "PDF report generation will be implemented in next version.\n\n"
            "Current workaround: Use browser print-to-PDF on the charts."
        )
        
    def export_csv(self):
        """Export results to CSV"""
        if not self.current_results:
            QMessageBox.warning(self, "Error", "No results to export")
            return
            
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Results",
            f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "CSV files (*.csv)"
        )
        
        if filename:
            try:
                # Export trades
                if 'trades' in self.current_results:
                    trades_df = pd.DataFrame(self.current_results['trades'])
                    trades_df.to_csv(filename, index=False)
                    QMessageBox.information(self, "Success", f"Exported to:\n{filename}")
                    self.status_update.emit("Results exported to CSV", "success")
                    
            except Exception as e:
                self.logger.error(f"Error exporting CSV: {e}")
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
                
    def export_trades(self):
        """Export filtered trades"""
        self.export_csv()
        
    def on_trade_double_clicked(self, item):
        """Handle double-click on trade"""
        row = item.row()
        
        # Collect trade data
        trade_data = {}
        for col in range(self.trade_table.columnCount()):
            header = self.trade_table.horizontalHeaderItem(col).text()
            cell_item = self.trade_table.item(row, col)
            trade_data[header] = cell_item.text() if cell_item else "N/A"
            
        # Emit signal (could open detail popup)
        self.trade_clicked.emit(trade_data)
        
        # Show simple message for now
        QMessageBox.information(
            self,
            f"Trade {row + 1} Details",
            "\n".join([f"{k}: {v}" for k, v in trade_data.items()])
        )
        
    def clear_display(self):
        """Clear all displays"""
        self.chart_view.setHtml("<div style='color:#cccccc; padding:20px;'><h3>No data loaded</h3><p>Run a backtest to see results</p></div>")
        self.trade_table.setRowCount(0)
        self.good_stats_table.setRowCount(0)
        self.bad_stats_table.setRowCount(0)
        self.recommendation_text.setText("Run a backtest to see recommendations")
        self.actions_text.setText("Actions will appear here based on results")
        self.current_result_text.setText("No backtest loaded")
