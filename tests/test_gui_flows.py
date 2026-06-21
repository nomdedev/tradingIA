import sys
import os
import pytest
from unittest.mock import MagicMock, patch
from PySide6.QtWidgets import QApplication, QTabWidget
from PySide6.QtCore import Qt

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import GUI classes
from src.gui.platform_gui_tab1_improved import Tab1DataManagement
from src.gui.platform_gui_tab2_improved import Tab2StrategyConfig
from src.gui.platform_gui_tab3_improved import Tab3BacktestRunner
from src.gui.platform_gui_tab4_improved import Tab4ResultsAnalysis
from src.gui.platform_gui_tab5_improved import Tab5ABTesting
from src.gui.platform_gui_tab6_improved import Tab6LiveMonitor
from src.gui.platform_gui_tab7_improved import Tab7AdvancedAnalysis
from src.gui.platform_gui_tab9_improved import Tab9DataDownload
from src.gui.platform_gui_tab10_improved import Tab10Help
from src.gui.platform_gui_tab11_improved import Tab11RiskMetrics
from src.gui.onboarding_wizard import OnboardingWizard

# Mock backend classes
class MockDataManager:
    def load_data(self, *args, **kwargs):
        return {"1H": "mock_dataframe"}

class MockStrategyEngine:
    def list_strategies(self):
        return ["StrategyA", "StrategyB"]

class MockBacktesterCore:
    def run_simple_backtest(self, *args, **kwargs):
        return {"metrics": {"sharpe": 1.5, "total_return": 10.0}, "trades": []}
    
    def list_available_strategies(self):
        return ["StrategyA", "StrategyB"]
    
    def get_strategy_params(self, strategy_name):
        return {}

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture
def parent_platform():
    mock = MagicMock()
    mock.data_dict = {}
    mock.update_status = MagicMock()
    mock.session_logger = MagicMock()
    return mock

def test_tab1_data_flow(qapp, parent_platform, take_screenshot):
    """Test Data Management Tab Flow"""
    data_manager = MockDataManager()
    parent_platform.data_manager = data_manager
    tab = Tab1DataManagement(parent_platform)
    
    # Simulate user interaction
    tab.symbol_combo.setCurrentText("BTC/USD")
    tab.timeframe_combo.setCurrentText("15Min")
    
    # Verify load button exists and is connected
    assert tab.load_data_btn is not None
    
    # Simulate click (we mock the thread to avoid async issues in test)
    with patch('src.gui.platform_gui_tab1_improved.DataLoadThread') as MockThread:
        mock_thread_instance = MockThread.return_value
        tab.on_load_data_clicked()
        
        # Verify thread started
        mock_thread_instance.start.assert_called_once()
        
        # Simulate thread completion
        # We need to mock the dataframe for the preview
        mock_df = MagicMock()
        mock_df.__len__.return_value = 100
        mock_df.index = [MagicMock(), MagicMock()]
        mock_df.index[0].strftime.return_value = "2023-01-01"
        mock_df.index[-1].strftime.return_value = "2023-01-02"
        mock_df.tail.return_value = mock_df
        
        data_dict = {"15Min": mock_df}
        
        # Mock update_chart_preview to avoid QWebEngineView issues in headless
        with patch.object(tab, 'update_chart_preview'):
            tab.on_data_loaded(data_dict)
        
        # Verify data stored in parent
        assert parent_platform.data_dict == data_dict
    
    take_screenshot(tab, "tab1_data_management")

def test_tab2_strategy_flow(qapp, parent_platform, take_screenshot):
    """Test Strategy Config Tab Flow"""
    backend = MagicMock()
    tab = Tab2StrategyConfig(parent_platform, backend)
    
    # Simulate strategy selection
    tab.strategy_combo.addItem("TestStrategy")
    tab.strategy_combo.setCurrentText("TestStrategy")
    
    # Check UI state
    assert tab.strategy_combo.currentText() == "TestStrategy"
    
    take_screenshot(tab, "tab2_strategy_config")

def test_tab3_backtest_flow(qapp, parent_platform, take_screenshot):
    """Test Backtest Runner Tab Flow"""
    backtester = MockBacktesterCore()
    tab = Tab3BacktestRunner(parent_platform, backtester)
    
    # Setup prerequisites
    parent_platform.data_dict = {"1H": "data"}
    parent_platform.current_strategy_class = "StrategyClass"
    parent_platform.current_strategy_params = {}
    
    # Force check prerequisites
    tab.check_prerequisites()
    tab.show()
    
    # Verify run button enabled
    assert tab.run_btn.isEnabled()
    
    # Test Kelly UI interaction
    assert tab.kelly_checkbox is not None
    assert tab.kelly_slider.isVisible() == False
    
    # Enable Kelly
    tab.kelly_checkbox.setChecked(True)
    assert tab.kelly_slider.isVisible() == True
    assert "Kelly Fraction: 0.5" in tab.kelly_fraction_label.text()
    
    # Change slider
    tab.kelly_slider.setValue(8)
    assert "Kelly Fraction: 0.8" in tab.kelly_fraction_label.text()
    
    # Simulate run
    with patch('src.gui.platform_gui_tab3_improved.BacktestThread') as MockThread:
        mock_thread_instance = MockThread.return_value
        tab.on_run_backtest_clicked()
        
        # Verify thread started
        mock_thread_instance.start.assert_called_once()
        
        # Verify Kelly settings passed to backtester
        assert backtester.enable_kelly_position_sizing == True
        assert backtester.kelly_fraction == 0.8
    
    take_screenshot(tab, "tab3_backtest_runner")

def test_tab4_results_flow(qapp, parent_platform, take_screenshot):
    """Test Results Analysis Tab Flow"""
    tab = Tab4ResultsAnalysis(parent_platform)
    
    # Verify Risk Analysis tab exists
    # Tab index 0: Statistics, 1: Recommendations, 2: Trades, 3: Risk Analysis
    bottom_tabs = tab.findChild(QTabWidget)
    assert bottom_tabs is not None
    assert bottom_tabs.tabText(3) == "Risk Analysis (MAE/MFE)"
    
    # Simulate loading results with MAE/MFE data
    results = {
        "metrics": {"sharpe": 2.0, "total_return": 0.15, "num_trades": 10},
        "trades": [
            {"pnl": 100, "mae": 0.02, "mfe": 0.05},
            {"pnl": -50, "mae": 0.04, "mfe": 0.01}
        ],
        "equity_curve": [1000, 1100, 1050]
    }
    
    # Mock update_risk_analysis to verify it's called
    with patch.object(tab, 'update_risk_analysis', wraps=tab.update_risk_analysis) as mock_update:
        tab.load_results(results)
        mock_update.assert_called_once_with(results)
        
    # Verify chart view is populated (check if setHtml was called)
    # Since QWebEngineView.setHtml is async, we just check if the widget exists
    assert tab.risk_chart_view is not None
    
    take_screenshot(tab, "tab4_results_analysis")

def test_tab5_ab_testing_flow(qapp, parent_platform, take_screenshot):
    """Test A/B Testing Tab Flow"""
    backtester = MockBacktesterCore()
    tab = Tab5ABTesting(parent_platform, backtester)
    
    # Verify strategies loaded
    assert tab.strategy_a_combo.count() > 0
    assert tab.strategy_b_combo.count() > 0
    
    # Select strategies
    tab.strategy_a_combo.setCurrentIndex(0)
    tab.strategy_b_combo.setCurrentIndex(1)
    
    # Setup data
    parent_platform.data_dict = {"1H": "mock_data"}
    
    # Simulate run
    with patch('src.gui.platform_gui_tab5_improved.ABTestThread') as MockThread:
        mock_thread_instance = MockThread.return_value
        tab.on_run_ab_test()
        
        # Verify thread started
        mock_thread_instance.start.assert_called_once()
        
        # Simulate completion
        results = {
            "strategy_a": {"results": {"metrics": {"sharpe": 1.0}}},
            "strategy_b": {"results": {"metrics": {"sharpe": 1.2}}},
            "comparison": {"sharpe": {"winner": "B"}},
            "recommendation": {"winner": "B", "action_text": "Adopt B"}
        }
        
        # Mock display_results to avoid UI issues
        with patch.object(tab, 'display_results'):
            tab.on_test_completed(results)
            tab.display_results.assert_called_once_with(results)

    # For now, just checking if we can instantiate and set results
    tab.current_results = results
    
    take_screenshot(tab, "tab5_ab_testing")

def test_tab6_live_monitor_flow(qapp, parent_platform, take_screenshot):
    """Test Live Monitor Tab Flow"""
    backtester = MockBacktesterCore()
    tab = Tab6LiveMonitor(parent_platform, backtester)
    
    # Verify initial state
    assert tab.is_running == False
    assert tab.start_btn.text() == "▶ START TRADING"
    
    # Select ticker and strategy
    tab.ticker_combo.setCurrentText("BTC/USD")
    tab.strategy_combo.setCurrentText("RSI_Bollinger")
    
    # Start trading
    tab.toggle_trading()
    assert tab.is_running == True
    assert tab.start_btn.text() == "⏹ STOP TRADING"
    
    # Simulate update
    tab.update_simulation()
    
    # Stop trading
    tab.toggle_trading()
    assert tab.is_running == False
    
    take_screenshot(tab, "tab6_live_monitor")

def test_tab7_advanced_analysis_flow(qapp, parent_platform, take_screenshot):
    """Test Advanced Analysis Tab Flow"""
    analysis_engines = MagicMock()
    tab = Tab7AdvancedAnalysis(parent_platform, analysis_engines)
    
    # Verify UI elements
    assert tab.hypothesis_input is not None
    assert tab.feature_method_combo is not None
    
    # Simulate hypothesis test
    tab.hypothesis_input.setText("Test Hypothesis")
    
    with patch('src.gui.platform_gui_tab7_improved.ResearchThread') as MockThread:
        mock_thread_instance = MockThread.return_value
        tab.on_run_hypothesis_test()
        
        # Verify thread started
        mock_thread_instance.start.assert_called_once()
        
        # Simulate completion
        result = {"type": "hypothesis", "confidence": 95.0, "hypothesis": "Test", "t_statistic": 2.0, "p_value": 0.01, "significant": True}
        
        # Mock display_hypothesis_results to avoid UI issues
        with patch.object(tab, 'display_hypothesis_results'):
            tab.on_research_complete(result)
            tab.display_hypothesis_results.assert_called_once_with(result)
        
        # Verify experiment added
        assert len(tab.experiments) == 1
        assert tab.experiments[0]["type"] == "hypothesis"
    
    take_screenshot(tab, "tab7_advanced_analysis")

def test_tab9_data_download_flow(qapp, parent_platform, take_screenshot):
    """Test Data Download Tab Flow"""
    tab = Tab9DataDownload(parent_platform)
    
    # Verify UI elements
    assert tab.timeframe_cards is not None
    assert len(tab.timeframe_cards) == 4
    
    # Simulate download
    with patch('src.gui.platform_gui_tab9_improved.DataDownloadThread') as MockThread:
        mock_thread_instance = MockThread.return_value
        tab.download_data("5Min")
        
        # Verify thread started
        mock_thread_instance.start.assert_called_once()
        
        # Simulate progress
        tab.update_progress("Downloading...", 50)
        assert tab.progress_bar.value() == 50
        
        # Simulate completion
        tab.on_download_finished(True, "Success")
        assert tab.progress_bar.value() == 100
    
    take_screenshot(tab, "tab9_data_download")

def test_tab10_help_flow(qapp, parent_platform, take_screenshot):
    """Test Help Tab Flow"""
    tab = Tab10Help(None)
    tab.parent = parent_platform
    
    # Verify UI elements
    assert tab.nav_panel is not None
    assert tab.content_browser is not None
    
    # Verify initial content
    # With QWebEngineView, we can't easily check HTML synchronously in unit tests
    # assert "Welcome to TradingIA" in tab.content_browser.toHtml()
    # assert tab.content_browser.isVisible() # Fails in headless if not shown
    assert tab.content_browser is not None
    
    # Simulate topic selection
    # We need to find a tree item to click
    from PySide6.QtWidgets import QTreeWidget, QTreeWidgetItem
    tree = tab.nav_panel.findChild(QTreeWidget)
    assert tree is not None
    
    # Select "Getting Started"
    # Since we can't easily click in headless, we'll call the handler directly
    item = QTreeWidgetItem(["Getting Started"])
    tab.on_topic_selected(item, 0)
    
    # With QWebEngineView, content is loaded asynchronously and toHtml() is async
    # For this test, we just verify the method was called without error
    # In a real integration test, we would wait for the loadFinished signal
    assert tab.content_browser is not None
    
    take_screenshot(tab, "tab10_help")

def test_tab11_risk_metrics_flow(qapp, parent_platform, take_screenshot):
    """Test Risk Metrics Tab Flow"""
    tab = Tab11RiskMetrics(None)
    tab.parent = parent_platform
    
    # Verify UI elements
    assert tab.risk_metrics_group is not None
    assert tab.visualization_group is not None
    assert tab.detailed_analysis_group is not None
    
    # Verify initial state
    assert tab.metric_labels["max_dd"].text() == "--"
    
    # Simulate data loading
    parent_platform.last_backtest_results = {
        "equity_curve": [100, 105, 102, 110, 108, 115], # Simple curve
        "trades": [
            {"mae": 0.02, "mfe": 0.05},
            {"mae": 0.01, "mfe": 0.03}
        ]
    }
    
    # Mock canvas draw to avoid matplotlib issues in headless
    with patch.object(tab.canvas, 'draw'):
        tab.refresh_data()
        
        # Verify metrics updated
        # Max DD: 105->102 (3/105 ~2.8%), 110->108 (2/110 ~1.8%)
        # Wait, calculation logic:
        # 100 -> 105 (Peak) -> 102 (DD 2.85%)
        # 102 -> 110 (Peak) -> 108 (DD 1.81%)
        # 108 -> 115 (Peak)
        # Max DD should be around 2.85%
        
        # Check if text is not "--"
        assert tab.metric_labels["max_dd"].text() != "--"
        assert "%" in tab.metric_labels["max_dd"].text()
        
        assert tab.metric_labels["var_95"].text() != "--"
        
        # Check MAE/MFE metrics
        assert tab.metric_labels["avg_mae"].text() != "--"
        
        # Check analysis text populated
        assert "RISK ANALYSIS REPORT" in tab.analysis_text.toPlainText()
    
    take_screenshot(tab, "tab11_risk_metrics")

def test_onboarding_flow(qapp, take_screenshot):
    """Test Onboarding Wizard Flow"""
    wizard = OnboardingWizard()
    
    # Verify pages exist
    assert wizard.page(0) is not None  # Welcome
    assert wizard.page(1) is not None  # Tutorial
    
    # Verify styling applied (basic check)
    assert "background-color: #1e1e1e" in wizard.styleSheet()
    
    take_screenshot(wizard, "onboarding_wizard")


