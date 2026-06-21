
import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PySide6.QtWidgets import QApplication, QWidget
from src.gui.styles import DarkTheme
from src.gui.platform_gui_tab1_improved import Tab1DataManagement
from src.gui.platform_gui_tab2_improved import Tab2StrategyConfig
from src.gui.platform_gui_tab3_improved import Tab3BacktestRunner
from src.gui.platform_gui_tab6_improved import Tab6LiveMonitor
from src.gui.platform_gui_tab10_improved import Tab10Help
from src.gui.platform_gui_tab11_improved import Tab11RiskMetrics

# Mock Parent Platform
class MockPlatform(QWidget):
    def __init__(self):
        super().__init__()
        self.data_manager = None
        self.backtester = None
        self.analysis_engines = None

def verify_ui_instantiation():
    print("Initializing QApplication...")
    app = QApplication(sys.argv)
    
    mock_platform = MockPlatform()
    
    print("Verifying Tab 1 (Data)...")
    try:
        tab1 = Tab1DataManagement(mock_platform)
        print("✅ Tab 1 instantiated successfully")
    except Exception as e:
        print(f"❌ Tab 1 failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verifying Tab 2 (Strategy)...")
    try:
        tab2 = Tab2StrategyConfig(mock_platform, None)
        print("✅ Tab 2 instantiated successfully")
    except Exception as e:
        print(f"❌ Tab 2 failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verifying Tab 3 (Backtest)...")
    try:
        tab3 = Tab3BacktestRunner(mock_platform, None)
        print("✅ Tab 3 instantiated successfully")
    except Exception as e:
        print(f"❌ Tab 3 failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verifying Tab 6 (Live)...")
    try:
        tab6 = Tab6LiveMonitor(mock_platform, None)
        print("✅ Tab 6 instantiated successfully")
    except Exception as e:
        print(f"❌ Tab 6 failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verifying Tab 10 (Help)...")
    try:
        tab10 = Tab10Help(mock_platform)
        print("✅ Tab 10 instantiated successfully")
    except Exception as e:
        print(f"❌ Tab 10 failed: {e}")
        import traceback
        traceback.print_exc()

    print("Verifying Tab 11 (Risk)...")
    try:
        tab11 = Tab11RiskMetrics(mock_platform)
        print("✅ Tab 11 instantiated successfully")
    except Exception as e:
        print(f"❌ Tab 11 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\nVerification Complete.")

if __name__ == "__main__":
    verify_ui_instantiation()
