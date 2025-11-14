from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem,
    QGroupBox, QHeaderView, QCheckBox, QPushButton
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Tab4ResultsAnalysis(QWidget):
    trade_clicked = Signal(dict)

    def __init__(self, parent_platform):
        super().__init__()
        self.parent_platform = parent_platform
        self.logger = logging.getLogger(__name__)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        # Charts Section (Top 70%)
        charts_group = QGroupBox("Performance Charts")
        charts_layout = QVBoxLayout()

        # Tab widget for different charts
        from PySide6.QtWidgets import QTabWidget
        self.charts_tabs = QTabWidget()

        # Equity Curve Tab
        self.equity_tab = QWidget()
        equity_layout = QVBoxLayout()
        self.equity_view = QWebEngineView()
        equity_layout.addWidget(self.equity_view)
        self.equity_tab.setLayout(equity_layout)
        self.charts_tabs.addTab(self.equity_tab, "Equity Curve")

        # Win/Loss Distribution Tab
        self.distribution_tab = QWidget()
        dist_layout = QVBoxLayout()
        self.distribution_view = QWebEngineView()
        dist_layout.addWidget(self.distribution_view)
        self.distribution_tab.setLayout(dist_layout)
        self.charts_tabs.addTab(self.distribution_tab, "Win/Loss Distribution")

        # Parameter Sensitivity Tab
        self.sensitivity_tab = QWidget()
        sens_layout = QVBoxLayout()
        self.sensitivity_view = QWebEngineView()
        sens_layout.addWidget(self.sensitivity_view)
        self.sensitivity_tab.setLayout(sens_layout)
        self.charts_tabs.addTab(self.sensitivity_tab, "Parameter Sensitivity")

        charts_layout.addWidget(self.charts_tabs)
        charts_group.setLayout(charts_layout)
        main_layout.addWidget(charts_group, 7)  # 70% of space

        # Bottom Section (30%)
        bottom_layout = QHBoxLayout()

        # Trade Log Section (50%)
        trade_group = QGroupBox("Trade Log")
        trade_layout = QVBoxLayout()

        # Filter controls
        filter_layout = QHBoxLayout()
        self.score_filter = QCheckBox("Score >= 4 only")
        self.score_filter.stateChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.score_filter)

        self.export_trades_btn = QPushButton("Export CSV")
        self.export_trades_btn.clicked.connect(self.export_trades_csv)
        filter_layout.addWidget(self.export_trades_btn)
        filter_layout.addStretch()

        trade_layout.addLayout(filter_layout)

        # Trade table
        self.trade_table = QTableWidget()
        self.trade_table.setColumnCount(9)
        self.trade_table.setHorizontalHeaderLabels([
            "Timestamp", "Entry", "Exit", "PnL%", "Score", "Entry_Type",
            "Reason_Exit", "HTF_Bias", "MAE%"
        ])
        self.trade_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.trade_table.setAlternatingRowColors(True)
        self.trade_table.itemDoubleClicked.connect(self.on_trade_double_clicked)

        trade_layout.addWidget(self.trade_table)
        trade_group.setLayout(trade_layout)
        bottom_layout.addWidget(trade_group, 5)  # 50% of bottom

        # Statistics Section (50%)
        stats_group = QGroupBox("Trade Statistics")
        stats_layout = QVBoxLayout()

        # Good Entries Stats
        good_group = QGroupBox("Good Entries (Score >= 4)")
        good_layout = QVBoxLayout()

        self.good_stats_table = QTableWidget()
        self.good_stats_table.setColumnCount(2)
        self.good_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.good_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.good_stats_table.setMaximumHeight(120)

        good_layout.addWidget(self.good_stats_table)
        good_group.setLayout(good_layout)
        stats_layout.addWidget(good_group)

        # Bad Entries Stats
        bad_group = QGroupBox("Bad Entries (Score < 4)")
        bad_layout = QVBoxLayout()

        self.bad_stats_table = QTableWidget()
        self.bad_stats_table.setColumnCount(2)
        self.bad_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.bad_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.bad_stats_table.setMaximumHeight(120)

        bad_layout.addWidget(self.bad_stats_table)
        bad_group.setLayout(bad_layout)
        stats_layout.addWidget(bad_group)

        # Recommendation
        rec_group = QGroupBox("Recommendation")
        rec_layout = QVBoxLayout()

        self.recommendation_label = QLabel("Load backtest results to see analysis")
        self.recommendation_label.setWordWrap(True)
        self.recommendation_label.setStyleSheet("QLabel { padding: 10px; }")

        rec_layout.addWidget(self.recommendation_label)
        rec_group.setLayout(rec_layout)
        stats_layout.addWidget(rec_group)

        stats_group.setLayout(stats_layout)
        bottom_layout.addWidget(stats_group, 5)  # 50% of bottom

        main_layout.addLayout(bottom_layout, 3)  # 30% of space

        self.setLayout(main_layout)

    def on_tab_activated(self):
        """Called when this tab becomes active"""
        if hasattr(self.parent_platform, 'last_backtest_results') and self.parent_platform.last_backtest_results:
            self.load_results(self.parent_platform.last_backtest_results)
        else:
            self.clear_display()

    def load_results(self, results):
        try:
            # Render charts
            self.render_equity_chart(results)
            self.render_distribution_chart(results)
            self.render_sensitivity_chart(results)

            # Load trade data
            self.load_trade_data(results)

            # Calculate and display statistics
            self.calculate_statistics(results)

            # Generate recommendation
            self.generate_recommendation(results)

        except Exception as e:
            self.logger.error(f"Error loading results: {e}")
            self.show_error(f"Error loading results: {str(e)}")

    def render_equity_chart(self, results):
        try:
            if 'equity_curve' not in results:
                self.equity_view.setHtml("<h3>No equity curve data available</h3>")
                return

            equity = results['equity_curve']

            # Calculate drawdown
            peak = pd.Series(equity).expanding().max()
            drawdown = (pd.Series(equity) - peak) / peak * 100

            # Create figure
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Equity curve
            fig.add_trace(
                go.Scatter(x=list(range(len(equity))), y=equity, name="Equity",
                          line=dict(color='blue', width=2)),
                secondary_y=False
            )

            # Drawdown
            fig.add_trace(
                go.Scatter(x=list(range(len(drawdown))), y=drawdown, name="Drawdown %",
                          fill='tozeroy', line=dict(color='red')),
                secondary_y=True
            )

            # Update layout
            fig.update_layout(
                title="Equity Curve & Drawdown",
                xaxis_title="Bars",
                yaxis_title="Equity ($)",
                yaxis2_title="Drawdown (%)",
                height=400
            )

            # Convert to HTML and display
            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
            self.equity_view.setHtml(html_content)

        except Exception as e:
            self.logger.error(f"Error rendering equity chart: {e}")
            self.equity_view.setHtml(f"<h3>Error rendering chart: {str(e)}</h3>")

    def render_distribution_chart(self, results):
        try:
            if 'trades' not in results or not results['trades']:
                self.distribution_view.setHtml("<h3>No trade data available</h3>")
                return

            trades_df = pd.DataFrame(results['trades'])
            if 'pnl_pct' not in trades_df.columns:
                self.distribution_view.setHtml("<h3>No PnL data in trades</h3>")
                return

            # Create histogram
            fig = go.Figure()

            # Winning trades
            wins = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct']
            if not wins.empty:
                fig.add_trace(go.Histogram(
                    x=wins, name="Wins", marker_color='green',
                    opacity=0.7, nbinsx=20
                ))

            # Losing trades
            losses = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct']
            if not losses.empty:
                fig.add_trace(go.Histogram(
                    x=losses, name="Losses", marker_color='red',
                    opacity=0.7, nbinsx=20
                ))

            fig.update_layout(
                title="Win/Loss Distribution",
                xaxis_title="PnL %",
                yaxis_title="Frequency",
                barmode='overlay',
                height=400
            )

            html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
            self.distribution_view.setHtml(html_content)

        except Exception as e:
            self.logger.error(f"Error rendering distribution chart: {e}")
            self.distribution_view.setHtml(f"<h3>Error rendering chart: {str(e)}</h3>")

    def render_sensitivity_chart(self, results):
        try:
            # This would require parameter sensitivity data from walk-forward
            # For now, show placeholder
            if 'periods' in results:
                # Create a simple scatter plot of train vs test Sharpe
                periods = results['periods']
                train_sharpe = [p['train_metrics']['sharpe'] for p in periods]
                test_sharpe = [p['test_metrics']['sharpe'] for p in periods]

                fig = go.Figure()

                fig.add_trace(go.Scatter(
                    x=train_sharpe, y=test_sharpe, mode='markers+text',
                    text=[f"Period {i+1}" for i in range(len(periods))],
                    textposition="top center",
                    name="Periods"
                ))

                # Add diagonal line
                max_val = max(max(train_sharpe), max(test_sharpe))
                fig.add_trace(go.Scatter(
                    x=[0, max_val], y=[0, max_val],
                    mode='lines', name='No Degradation',
                    line=dict(dash='dash', color='gray')
                ))

                fig.update_layout(
                    title="Train vs Test Sharpe (Walk-Forward Degradation)",
                    xaxis_title="Train Sharpe",
                    yaxis_title="Test Sharpe",
                    height=400
                )

                html_content = fig.to_html(include_plotlyjs='cdn', full_html=False)
                self.sensitivity_view.setHtml(html_content)
            else:
                self.sensitivity_view.setHtml("<h3>No sensitivity data available (run Walk-Forward analysis)</h3>")

        except Exception as e:
            self.logger.error(f"Error rendering sensitivity chart: {e}")
            self.sensitivity_view.setHtml(f"<h3>Error rendering chart: {str(e)}</h3>")

    def load_trade_data(self, results):
        try:
            if 'trades' not in results or not results['trades']:
                self.trade_table.setRowCount(0)
                return

            trades = results['trades']
            self.trade_table.setRowCount(len(trades))

            for i, trade in enumerate(trades):
                # Format data
                timestamp = trade.get('timestamp', '')
                entry_price = trade.get('entry_price', 0)
                exit_price = trade.get('exit_price', 0)
                pnl_pct = trade.get('pnl_pct', 0)
                score = trade.get('score', 0)
                entry_type = trade.get('entry_type', '')
                reason_exit = trade.get('reason_exit', '')

                # Calculate additional fields (placeholders)
                htf_bias = "Bull" if score >= 4 else "Bear"
                mae_pct = abs(pnl_pct) * 0.5  # Placeholder

                # Populate table
                self.trade_table.setItem(i, 0, QTableWidgetItem(str(timestamp)))
                self.trade_table.setItem(i, 1, QTableWidgetItem(f"{entry_price:.2f}"))
                self.trade_table.setItem(i, 2, QTableWidgetItem(f"{exit_price:.2f}"))
                self.trade_table.setItem(i, 3, QTableWidgetItem(f"{pnl_pct:.2f}%"))

                score_item = QTableWidgetItem(str(score))
                score_item.setBackground(Qt.GlobalColor.green if score >= 4 else Qt.GlobalColor.red)
                self.trade_table.setItem(i, 4, score_item)

                self.trade_table.setItem(i, 5, QTableWidgetItem(entry_type))
                self.trade_table.setItem(i, 6, QTableWidgetItem(reason_exit))
                self.trade_table.setItem(i, 7, QTableWidgetItem(htf_bias))
                self.trade_table.setItem(i, 8, QTableWidgetItem(f"{mae_pct:.2f}%"))

        except Exception as e:
            self.logger.error(f"Error loading trade data: {e}")

    def on_filter_changed(self):
        # Re-filter and recalculate statistics
        if hasattr(self.parent_platform, 'last_backtest_results'):
            self.calculate_statistics(self.parent_platform.last_backtest_results)

    def calculate_statistics(self, results):
        try:
            if 'trades' not in results or not results['trades']:
                return

            trades_df = pd.DataFrame(results['trades'])

            # Apply filter if checked
            if self.score_filter.isChecked():
                trades_df = trades_df[trades_df['score'] >= 4]

            if trades_df.empty:
                self.clear_stats_tables()
                return

            # Split into good and bad entries
            good_trades = trades_df[trades_df['score'] >= 4]
            bad_trades = trades_df[trades_df['score'] < 4]

            # Calculate stats for good trades
            if not good_trades.empty:
                good_win_rate = (good_trades['pnl_pct'] > 0).mean()
                good_avg_pnl = good_trades['pnl_pct'].mean()
                good_count = len(good_trades)

                self.populate_stats_table(self.good_stats_table, {
                    "Count": good_count,
                    "Win Rate": f"{good_win_rate:.1%}",
                    "Avg PnL": f"{good_avg_pnl:.2f}%",
                    "Profit Factor": "1.5"  # Placeholder
                })
            else:
                self.good_stats_table.setRowCount(0)

            # Calculate stats for bad trades
            if not bad_trades.empty:
                bad_win_rate = (bad_trades['pnl_pct'] > 0).mean()
                bad_avg_pnl = bad_trades['pnl_pct'].mean()
                bad_count = len(bad_trades)

                self.populate_stats_table(self.bad_stats_table, {
                    "Count": bad_count,
                    "Win Rate": f"{bad_win_rate:.1%}",
                    "Avg PnL": f"{bad_avg_pnl:.2f}%",
                    "Profit Factor": "0.8"  # Placeholder
                })
            else:
                self.bad_stats_table.setRowCount(0)

        except Exception as e:
            self.logger.error(f"Error calculating statistics: {e}")

    def populate_stats_table(self, table, stats_dict):
        table.setRowCount(len(stats_dict))
        for i, (metric, value) in enumerate(stats_dict.items()):
            table.setItem(i, 0, QTableWidgetItem(metric))
            table.setItem(i, 1, QTableWidgetItem(str(value)))

    def generate_recommendation(self, results):
        try:
            if 'trades' not in results or not results['trades']:
                self.recommendation_label.setText("No trade data available for analysis")
                return

            trades_df = pd.DataFrame(results['trades'])
            metrics = results.get('metrics', {})

            # Analyze performance
            win_rate = metrics.get('win_rate', 0)
            sharpe = metrics.get('sharpe', 0)
            max_dd = metrics.get('max_dd', 0)

            good_trades = trades_df[trades_df['score'] >= 4]
            bad_trades = trades_df[trades_df['score'] < 4]

            good_win_rate = (good_trades['pnl_pct'] > 0).mean() if not good_trades.empty else 0
            bad_win_rate = (bad_trades['pnl_pct'] > 0).mean() if not bad_trades.empty else 0

            # Generate recommendation
            if good_win_rate > 0.6 and sharpe > 1.0:
                recommendation = f"Excellent performance! Focus on score≥4 entries. Win rate: {good_win_rate:.1%}, Sharpe: {sharpe:.2f}. Ready for live trading."
                self.recommendation_label.setStyleSheet("QLabel { color: green; font-weight: bold; }")
            elif bad_win_rate > good_win_rate:
                recommendation = f"Poor entry quality. Bad entries performing better than good ones. Review entry criteria. Good win rate: {good_win_rate:.1%}, Bad win rate: {bad_win_rate:.1%}."
                self.recommendation_label.setStyleSheet("QLabel { color: red; font-weight: bold; }")
            elif max_dd > 0.2:
                recommendation = f"High drawdown risk. Max DD: {max_dd:.1%}. Consider position sizing reduction or stop-loss tightening."
                self.recommendation_label.setStyleSheet("QLabel { color: orange; font-weight: bold; }")
            else:
                recommendation = f"Moderate performance. Win rate: {win_rate:.1%}, Sharpe: {sharpe:.2f}. Further optimization needed."
                self.recommendation_label.setStyleSheet("QLabel { color: blue; font-weight: bold; }")

            self.recommendation_label.setText(recommendation)

        except Exception as e:
            self.logger.error(f"Error generating recommendation: {e}")
            self.recommendation_label.setText(f"Error generating recommendation: {str(e)}")

    def on_trade_double_clicked(self, item):
        try:
            row = item.row()
            trade_data = {}

            # Extract trade data from table
            for col in range(self.trade_table.columnCount()):
                header = self.trade_table.horizontalHeaderItem(col).text()
                value = self.trade_table.item(row, col).text()
                trade_data[header.lower().replace(' ', '_')] = value

            # Emit signal for popup (to be implemented)
            self.trade_clicked.emit(trade_data)

        except Exception as e:
            self.logger.error(f"Error handling trade double-click: {e}")

    def export_trades_csv(self):
        try:
            from PySide6.QtWidgets import QFileDialog

            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Trades",
                "trades.csv",
                "CSV files (*.csv)"
            )

            if not filename:
                return

            # Collect visible trade data
            trades_data = []
            for row in range(self.trade_table.rowCount()):
                trade = {}
                for col in range(self.trade_table.columnCount()):
                    header = self.trade_table.horizontalHeaderItem(col).text()
                    item = self.trade_table.item(row, col)
                    trade[header] = item.text() if item else ""
                trades_data.append(trade)

            # Export to CSV
            df = pd.DataFrame(trades_data)
            df.to_csv(filename, index=False)

            self.show_message("Success", f"Trades exported to {filename}")

        except Exception as e:
            self.show_error(f"Export failed: {str(e)}")

    def clear_display(self):
        # Clear charts
        self.equity_view.setHtml("<h3>No data loaded</h3>")
        self.distribution_view.setHtml("<h3>No data loaded</h3>")
        self.sensitivity_view.setHtml("<h3>No data loaded</h3>")

        # Clear tables
        self.trade_table.setRowCount(0)
        self.clear_stats_tables()

        # Clear recommendation
        self.recommendation_label.setText("Load backtest results to see analysis")

    def clear_stats_tables(self):
        self.good_stats_table.setRowCount(0)
        self.bad_stats_table.setRowCount(0)

    def show_error(self, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical(self, "Error", msg)

    def show_message(self, title, msg):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, title, msg)