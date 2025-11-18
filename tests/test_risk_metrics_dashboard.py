#!/usr/bin/env python3
"""
Test script for Risk Metrics Dashboard functionality
Tests all features of Tab11RiskMetrics systematically
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test 1: Import all required modules"""
    from src.main_platform import TradingPlatform
    from src.gui.platform_gui_tab11_risk_metrics import Tab11RiskMetrics
    from core.execution.backtester_core import BacktesterCore
    from core.backend_core import DataManager, StrategyEngine
    # If we reach here without exception, imports are successful
    assert True

@pytest.fixture
def mock_results():
    """Test 2: Create mock backtest data for testing"""
    print("\n🧪 TEST 2: Creating mock backtest data...")

    try:
        # Create mock returns data
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        np.random.seed(42)  # For reproducible results

        # Generate realistic returns (mean ~0.05%, std ~2%)
        returns = np.random.normal(0.0005, 0.02, len(dates))

        # Add some drawdown periods
        drawdown_period = slice(100, 150)
        returns[drawdown_period] = np.random.normal(-0.005, 0.03, len(returns[drawdown_period]))

        # Create mock trades data
        trades = []
        capital = 10000
        position_size = 0.1  # 10% position size

        for i in range(50):  # 50 trades
            entry_date = dates[np.random.randint(0, len(dates)-5)]
            exit_date = entry_date + timedelta(days=np.random.randint(1, 5))

            # Simulate trade
            entry_price = 100 + np.random.normal(0, 5)
            pnl_pct = np.random.normal(0.02, 0.05)  # Mean 2%, std 5%
            exit_price = entry_price * (1 + pnl_pct)

            # Calculate MAE/MFE (simplified)
            mae = abs(pnl_pct) * np.random.uniform(0.3, 0.8)  # 30-80% of total move
            mfe = abs(pnl_pct) * np.random.uniform(1.2, 2.0)  # 120-200% of total move

            trade = {
                'entry_date': entry_date,
                'exit_date': exit_date,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': capital * position_size * pnl_pct,
                'pnl_pct': pnl_pct,
                'mae': mae,
                'mfe': mfe,
                'side': 'buy' if pnl_pct > 0 else 'sell'
            }
            trades.append(trade)

        # Calculate metrics
        total_return = (1 + returns).prod() - 1
        
        # Sharpe Ratio correcto: restar risk-free rate
        rf_daily = 0.04 / 252  # Risk-free rate diario
        excess_returns = returns - rf_daily
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0.0
        
        max_drawdown = 0
        peak = 1
        for ret in (1 + returns).cumprod():
            if ret > peak:
                peak = ret
            drawdown = (peak - ret) / peak
            max_drawdown = max(max_drawdown, drawdown)

        # Create mock results
        mock_results = {
            'metrics': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': 0.58,
                'profit_factor': 1.35,
                'var_95': -0.025,
                'expected_shortfall': -0.035,
                'sortino_ratio': sharpe_ratio * 0.8,
                'calmar_ratio': total_return / max_drawdown
            },
            'returns': returns.tolist(),
            'trades': trades,
            'mae_mfe': {
                'avg_mae': np.mean([t['mae'] for t in trades]),
                'avg_mfe': np.mean([t['mfe'] for t in trades]),
                'ratio': np.mean([t['mfe'] for t in trades]) / np.mean([t['mae'] for t in trades]),
                'max_mae': max([t['mae'] for t in trades]),
                'max_mfe': max([t['mfe'] for t in trades]),
                'recovery_factor': total_return / max_drawdown,
                'mae_values': [t['mae'] for t in trades],
                'mfe_values': [t['mfe'] for t in trades]
            },
            'risk_events': [
                {
                    'date': '2023-05-15',
                    'type': 'Drawdown',
                    'severity': 'Medium',
                    'impact': -0.08,
                    'description': 'Drawdown exceeded 8% threshold'
                },
                {
                    'date': '2023-08-22',
                    'type': 'MAE',
                    'severity': 'Low',
                    'impact': -0.045,
                    'description': 'High MAE on single trade'
                }
            ]
        }

        print("✅ Mock backtest data created successfully")
        print(f"   - {len(returns)} daily returns")
        print(f"   - {len(trades)} trades")
        print(f"   - Total return: {total_return:.3f}")
        print(f"   - Sharpe ratio: {sharpe_ratio:.3f}")
        print(f"   - Max drawdown: {max_drawdown:.3f}")
        return mock_results

    except Exception as e:
        pytest.fail(f"Failed to create mock data: {e}")

def test_risk_metrics_calculation(mock_results):
    """Test 3: Test risk metrics calculation"""
    # Test basic metrics
    metrics = mock_results['metrics']

    required_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown',
                      'win_rate', 'profit_factor', 'var_95', 'expected_shortfall']

    for metric in required_metrics:
        assert metric in metrics, f"Missing metric: {metric}"

    # Test MAE/MFE calculations
    mae_mfe = mock_results['mae_mfe']
    required_mae_mfe = ['avg_mae', 'avg_mfe', 'ratio', 'max_mae', 'max_mfe']

    for metric in required_mae_mfe:
        assert metric in mae_mfe, f"Missing MAE/MFE metric: {metric}"

class MockBacktester:
    """Mock backtester for testing"""
    def __init__(self, results):
        self.last_results = results

@pytest.fixture
def dashboard(mock_results):
    """Fixture to create dashboard instance"""
    from PySide6.QtWidgets import QApplication, QWidget
    import sys
    from src.gui.platform_gui_tab11_risk_metrics import Tab11RiskMetrics

    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create mock parent (QWidget)
    parent = QWidget()

    # Add backtester attribute to parent
    parent.backtester = MockBacktester(mock_results)

    # Create dashboard
    dashboard_instance = Tab11RiskMetrics(parent)
    
    yield dashboard_instance
    
    # Cleanup
    try:
        parent.close()
    except:
        pass

def test_risk_dashboard_initialization(dashboard):
    """Test 4: Test Risk Dashboard initialization"""
    # Test UI components exist
    assert hasattr(dashboard, 'risk_metrics_group'), "Risk metrics group not found"
    assert hasattr(dashboard, 'visualization_group'), "Visualization group not found"
    assert hasattr(dashboard, 'detailed_analysis_group'), "Detailed analysis group not found"

    # Test metric labels exist
    expected_labels = ['max_dd', 'var_95', 'es_95', 'sharpe', 'sortino', 'calmar',
                      'avg_mae', 'avg_mfe', 'mae_mfe_ratio', 'max_mae', 'max_mfe', 'recovery_factor']

    for label_key in expected_labels:
        assert label_key in dashboard.metric_labels, f"Missing metric label: {label_key}"

def test_data_loading(dashboard, mock_results):
    """Test 5: Test data loading and display"""
    # Load data
    dashboard.update_risk_metrics(mock_results)

    # Check if metrics are displayed
    metrics_updated = 0
    for label_key, label in dashboard.metric_labels.items():
        if label.text() != "--":
            metrics_updated += 1

    # At least some metrics should be updated
    assert metrics_updated > 0, "No metrics were updated"

    # Check status
    assert "updated successfully" in dashboard.status_label.text().lower(), "Status not updated"

def test_chart_generation(dashboard):
    """Test 6: Test chart generation"""
    chart_types = [
        "MAE/MFE Distribution",
        "Drawdown Analysis",
        "Volatility Clustering",
        "Stress Test Scenarios",
        "Risk-Return Scatter",
        "Tail Risk Analysis"
    ]

    for chart_type in chart_types:
        dashboard.chart_combo.setCurrentText(chart_type)
        dashboard.update_chart()

        # Check if figure was updated
        assert len(dashboard.figure.get_axes()) > 0, f"Chart not generated for {chart_type}"

def test_stress_testing(dashboard):
    """Test 7: Test stress testing functionality"""
    # Run stress test
    dashboard.run_stress_test()

    # Check if stress test results were generated
    assert hasattr(dashboard, 'stress_test_results'), "Stress test results not generated"
    assert dashboard.stress_test_results is not None, "Stress test results are None"

    scenarios = dashboard.stress_test_results.get('scenarios', {})
    assert len(scenarios) > 0, "No stress scenarios generated"

    # Check specific scenarios
    expected_scenarios = ['Market Crash (-20%)', 'Flash Crash (-10%)',
                        'High Volatility (+50%)', 'Liquidity Crisis', 'Interest Rate Hike']

    for scenario in expected_scenarios:
        assert scenario in scenarios, f"Missing scenario: {scenario}"
        assert 'loss' in scenarios[scenario], f"No loss calculated for {scenario}"

def test_risk_events_table(dashboard, mock_results):
    """Test 8: Test risk events table"""
    dashboard.update_risk_events_table(mock_results)

    # Check if table has rows
    row_count = dashboard.risk_table.rowCount()
    assert row_count > 0, "No risk events in table"

    # Check table structure
    assert dashboard.risk_table.columnCount() == 5, "Wrong number of columns"

    headers = ['Date', 'Type', 'Severity', 'Impact', 'Description']
    for i, header in enumerate(headers):
        assert dashboard.risk_table.horizontalHeaderItem(i).text() == header, f"Wrong header: {header}"

def test_analysis_report(dashboard, mock_results):
    """Test 9: Test analysis report generation"""
    dashboard.update_analysis_report(mock_results)

    # Check if report was generated
    report_text = dashboard.analysis_text.toPlainText()
    assert len(report_text) > 0, "Analysis report is empty"

    # Check report contains expected sections that are actually generated
    expected_sections = ['PERFORMANCE METRICS', 'MAE/MFE ANALYSIS', 'RISK RECOMMENDATIONS']
    for section in expected_sections:
        assert section in report_text, f"Missing section: {section}"

def run_all_tests():
    """Run all tests systematically"""
    print("🚀 INICIANDO PRUEBAS DEL RISK METRICS DASHBOARD")
    print("=" * 60)

    test_results = []

    # Test 1: Imports
    test_results.append(("Imports", test_imports()))

    # Test 2: Mock Data
    mock_data = test_mock_backtest_data()
    test_results.append(("Mock Data Creation", mock_data is not None))

    if mock_data:
        # Test 3: Risk Metrics Calculation
        test_results.append(("Risk Metrics Calculation", test_risk_metrics_calculation(mock_data)))

        # Test 4: Dashboard Initialization
        dashboard = test_risk_dashboard_initialization(mock_data)
        test_results.append(("Dashboard Initialization", dashboard is not None))

        if dashboard:
            # Test 5: Data Loading
            test_results.append(("Data Loading", test_data_loading(dashboard, mock_data)))

            # Test 6: Chart Generation
            test_results.append(("Chart Generation", test_chart_generation(dashboard)))

            # Test 7: Stress Testing
            test_results.append(("Stress Testing", test_stress_testing(dashboard)))

            # Test 8: Risk Events Table
            test_results.append(("Risk Events Table", test_risk_events_table(dashboard, mock_data)))

            # Test 9: Analysis Report
            test_results.append(("Analysis Report", test_analysis_report(dashboard, mock_data)))

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)

    passed = 0
    total = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print("30")
        if result:
            passed += 1

    print(f"\nResultado Final: {passed}/{total} pruebas pasaron")

    if passed == total:
        print("🎉 ¡TODAS LAS PRUEBAS PASARON! El Risk Metrics Dashboard está funcionando correctamente.")
        return True
    else:
        print("⚠️  Algunas pruebas fallaron. Revisar los errores arriba.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)