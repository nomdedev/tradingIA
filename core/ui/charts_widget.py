"""
Charts Widget - Interactive charts for market data and backtest results
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QGroupBox, QSplitter, QTableWidget, QTableWidgetItem,
    QHeaderView, QProgressBar, QTextEdit, QTabWidget
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QDateTimeAxis,
    QScatterSeries, QAreaSeries, QBarSeries, QBarSet
)
from PySide6.QtGui import QPainter, QColor, QPen, QBrush

logger = logging.getLogger(__name__)

class ChartsWidget(QWidget):
    """
    Interactive charts widget for displaying market data and backtest results.
    """

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.current_data: Optional[pd.DataFrame] = None
        self.backtest_results: List[Any] = []
        self.setup_ui()

    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)

        # Chart type selector
        chart_controls = QHBoxLayout()

        self.chart_type_combo = QComboBox()
        self.chart_type_combo.addItems([
            "Price Chart", "Volume Chart", "Equity Curve",
            "Returns Distribution", "Drawdown Chart", "Strategy Comparison"
        ])
        self.chart_type_combo.currentTextChanged.connect(self.update_chart)
        chart_controls.addWidget(QLabel("Chart Type:"))
        chart_controls.addWidget(self.chart_type_combo)

        chart_controls.addStretch()
        layout.addLayout(chart_controls)

        # Chart view
        self.chart_view = QChartView()
        self.chart_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        layout.addWidget(self.chart_view)

        # Chart info panel
        self.info_text = QTextEdit()
        self.info_text.setMaximumHeight(100)
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text)

    def set_data(self, data: pd.DataFrame):
        """Set market data for charting"""
        self.current_data = data
        self.update_chart()

    def set_backtest_results(self, results: List[Any]):
        """Set backtest results for charting"""
        self.backtest_results = results
        self.update_chart()

    def update_chart(self):
        """Update the chart based on current selection"""
        chart_type = self.chart_type_combo.currentText()

        if chart_type == "Price Chart":
            self._create_price_chart()
        elif chart_type == "Volume Chart":
            self._create_volume_chart()
        elif chart_type == "Equity Curve":
            self._create_equity_chart()
        elif chart_type == "Returns Distribution":
            self._create_returns_distribution()
        elif chart_type == "Drawdown Chart":
            self._create_drawdown_chart()
        elif chart_type == "Strategy Comparison":
            self._create_strategy_comparison()

    def _create_price_chart(self):
        """Create candlestick-style price chart"""
        if self.current_data is None or len(self.current_data) == 0:
            self._show_empty_chart("No market data available")
            return

        chart = QChart()
        chart.setTitle("Price Chart")

        # Create price series
        price_series = QLineSeries()
        price_series.setName("Close Price")

        for i, (date, row) in enumerate(self.current_data.iterrows()):
            timestamp = date.toPyDateTime().timestamp() * 1000  # Convert to milliseconds
            price_series.append(timestamp, row['close'])

        chart.addSeries(price_series)

        # Setup axes
        axis_x = QDateTimeAxis()
        axis_x.setFormat("MMM yyyy")
        axis_x.setTitleText("Date")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        price_series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Price")
        min_price = self.current_data['low'].min()
        max_price = self.current_data['high'].max()
        axis_y.setRange(min_price * 0.95, max_price * 1.05)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        price_series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self._update_info("Price chart showing closing prices over time")

    def _create_volume_chart(self):
        """Create volume chart"""
        if self.current_data is None or len(self.current_data) == 0:
            self._show_empty_chart("No market data available")
            return

        chart = QChart()
        chart.setTitle("Volume Chart")

        # Create volume bars
        volume_set = QBarSet("Volume")
        volume_data = self.current_data['volume'].values

        # Sample every 10th point for performance
        step = max(1, len(volume_data) // 100)
        for i in range(0, len(volume_data), step):
            volume_set.append(volume_data[i])

        volume_series = QBarSeries()
        volume_series.append(volume_set)

        chart.addSeries(volume_series)

        # Setup axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Period")
        axis_x.setRange(0, len(volume_data) // step)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        volume_series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Volume")
        axis_y.setRange(0, volume_data.max() * 1.1)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        volume_series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self._update_info("Volume chart showing trading volume over time")

    def _create_equity_chart(self):
        """Create equity curve chart from backtest results"""
        if not self.backtest_results:
            self._show_empty_chart("No backtest results available")
            return

        chart = QChart()
        chart.setTitle("Equity Curves")

        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                 QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255)]

        for i, result in enumerate(self.backtest_results):
            if result.equity_curve is not None:
                series = QLineSeries()
                series.setName(f"Strategy {i+1}")
                series.setColor(colors[i % len(colors)])

                equity_data = result.equity_curve.values
                for j, equity in enumerate(equity_data):
                    series.append(j, equity)

                chart.addSeries(series)

        # Setup axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Period")
        axis_x.setRange(0, len(self.backtest_results[0].equity_curve) if self.backtest_results else 100)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

        axis_y = QValueAxis()
        axis_y.setTitleText("Equity")
        # Set range based on all equity curves
        if self.backtest_results:
            all_equities = []
            for result in self.backtest_results:
                if result.equity_curve is not None:
                    all_equities.extend(result.equity_curve.values)
            if all_equities:
                axis_y.setRange(min(all_equities) * 0.95, max(all_equities) * 1.05)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        # Attach all series to axes
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self._update_info(f"Equity curves for {len(self.backtest_results)} strategies")

    def _create_returns_distribution(self):
        """Create returns distribution histogram"""
        if not self.backtest_results:
            self._show_empty_chart("No backtest results available")
            return

        chart = QChart()
        chart.setTitle("Returns Distribution")

        # Collect all returns
        all_returns = []
        for result in self.backtest_results:
            if result.equity_curve is not None:
                returns = result.equity_curve.pct_change().dropna().values
                all_returns.extend(returns)

        if not all_returns:
            self._show_empty_chart("No returns data available")
            return

        # Create histogram
        returns_series = QLineSeries()
        returns_series.setName("Returns Distribution")

        # Simple histogram approximation
        hist, bins = np.histogram(all_returns, bins=20)
        for i in range(len(hist)):
            returns_series.append(bins[i], hist[i])

        chart.addSeries(returns_series)

        # Setup axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Return")
        axis_x.setRange(min(all_returns), max(all_returns))
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)
        returns_series.attachAxis(axis_x)

        axis_y = QValueAxis()
        axis_y.setTitleText("Frequency")
        axis_y.setRange(0, max(hist) * 1.1)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)
        returns_series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self._update_info("Distribution of returns across all strategies")

    def _create_drawdown_chart(self):
        """Create drawdown chart"""
        if not self.backtest_results:
            self._show_empty_chart("No backtest results available")
            return

        chart = QChart()
        chart.setTitle("Drawdown Analysis")

        colors = [QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255)]

        for i, result in enumerate(self.backtest_results[:3]):  # Show first 3
            if result.equity_curve is not None:
                series = QLineSeries()
                series.setName(f"Strategy {i+1} Drawdown")
                series.setColor(colors[i])

                equity = result.equity_curve.values
                peak = equity[0]
                for j, value in enumerate(equity):
                    peak = max(peak, value)
                    drawdown = (value - peak) / peak
                    series.append(j, drawdown)

                chart.addSeries(series)

        # Setup axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Period")
        axis_x.setRange(0, len(self.backtest_results[0].equity_curve) if self.backtest_results else 100)
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

        axis_y = QValueAxis()
        axis_y.setTitleText("Drawdown")
        axis_y.setRange(-0.5, 0)  # Drawdown is always negative
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        # Attach all series to axes
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self._update_info("Drawdown analysis showing maximum losses over time")

    def _create_strategy_comparison(self):
        """Create strategy comparison chart"""
        if not self.backtest_results:
            self._show_empty_chart("No backtest results available")
            return

        chart = QChart()
        chart.setTitle("Strategy Performance Comparison")

        # Extract key metrics
        strategies = []
        sharpe_ratios = []
        total_returns = []
        max_drawdowns = []

        for i, result in enumerate(self.backtest_results):
            metrics = result.metrics
            strategies.append(f"Strategy {i+1}")
            sharpe_ratios.append(metrics.get('sharpe_ratio', 0))
            total_returns.append(metrics.get('total_return', 0))
            max_drawdowns.append(abs(metrics.get('max_drawdown', 0)))

        # Create bar series for each metric
        if sharpe_ratios:
            sharpe_set = QBarSet("Sharpe Ratio")
            for ratio in sharpe_ratios:
                sharpe_set.append(ratio)

            sharpe_series = QBarSeries()
            sharpe_series.append(sharpe_set)
            chart.addSeries(sharpe_series)

        # Setup axes
        axis_x = QValueAxis()
        axis_x.setTitleText("Strategy")
        axis_x.setRange(0, len(strategies))
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

        axis_y = QValueAxis()
        axis_y.setTitleText("Value")
        if sharpe_ratios:
            axis_y.setRange(min(sharpe_ratios) * 0.9, max(sharpe_ratios) * 1.1)
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        # Attach series to axes
        for series in chart.series():
            series.attachAxis(axis_x)
            series.attachAxis(axis_y)

        self.chart_view.setChart(chart)
        self._update_info("Strategy comparison showing Sharpe ratios")

    def _show_empty_chart(self, message: str):
        """Show empty chart with message"""
        chart = QChart()
        chart.setTitle(message)
        self.chart_view.setChart(chart)
        self._update_info(message)

    def _update_info(self, text: str):
        """Update the info text area"""
        self.info_text.setPlainText(text)