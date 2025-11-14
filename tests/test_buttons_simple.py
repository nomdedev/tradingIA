#!/usr/bin/env python3
"""
Script simplificado para probar botones de cada tab individualmente
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QPushButton, QWidget

# Create QApplication first
app = QApplication(sys.argv)

# Mock classes
class MockDataManager:
    def load_alpaca_data(self, *args, **kwargs):
        return {"error": "Mock data manager"}

class MockStrategyEngine:
    pass

class MockBacktester:
    def list_available_strategies(self):
        return []

class MockAnalysisEngines:
    pass

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_tab_buttons(tab_class, tab_name, *args, **kwargs):
    """Test buttons in a specific tab class"""
    print(f"\n🔍 Testing Tab: {tab_name}")

    try:
        # Create a mock parent for tabs that need it
        class MockParent(QWidget):
            def __init__(self):
                super().__init__()
                self.tabs = None
                self.data_dict = {}
                self.config_dict = {}
                self.last_backtest_results = {}
                # Add data_manager mock
                self.data_manager = MockDataManager()

        mock_parent = MockParent()

        # Create tab instance
        if args or kwargs:
            tab = tab_class(mock_parent, *args, **kwargs)
        else:
            tab = tab_class(mock_parent)

        # Find all QPushButton instances
        buttons = tab.findChildren(QPushButton)
        print(f"   Found {len(buttons)} buttons")

        working_buttons = []
        broken_buttons = []

        for button in buttons:
            button_text = button.text().strip()
            if not button_text:
                button_text = button.objectName() or "Unnamed Button"

            try:
                # Check if button has a click handler
                if hasattr(button, 'clicked') and button.clicked:
                    working_buttons.append(button_text)
                    print(f"   ✅ '{button_text}' - Has click handler")
                else:
                    working_buttons.append(button_text)
                    print(f"   ⚠️  '{button_text}' - No click handler attached")

            except Exception as e:
                broken_buttons.append((button_text, str(e)))
                print(f"   ❌ '{button_text}' - Error: {str(e)}")

        return working_buttons, broken_buttons

    except Exception as e:
        print(f"   ❌ Failed to create tab: {str(e)}")
        return [], [(tab_name, str(e))]

def main():
    print("🚀 Testing Trading IA Platform Buttons")
    print("=" * 60)

    # Import tab classes
    from gui.platform_gui_tab0 import Tab0Dashboard
    from gui.platform_gui_tab1_improved import Tab1DataManagement
    from gui.platform_gui_tab2_improved import Tab2StrategyConfig
    from gui.platform_gui_tab3_improved import Tab3BacktestRunner
    from gui.platform_gui_tab4_improved import Tab4ResultsAnalysis
    from gui.platform_gui_tab5_improved import Tab5ABTesting
    from gui.platform_gui_tab6_improved import Tab6LiveMonitoring
    from gui.platform_gui_tab7_improved import Tab7AdvancedAnalysis
    from gui.platform_gui_tab8 import Tab8SystemSettings
    from gui.platform_gui_tab9_data_download import Tab9DataDownload
    from gui.platform_gui_tab10_help import Tab10Help

    # Mock backend classes
    class MockStrategyEngine:
        pass

    class MockBacktester:
        pass

    class MockAnalysisEngines:
        pass

    # Test each tab
    tabs_to_test = [
        (Tab0Dashboard, "Dashboard"),
        (Tab1DataManagement, "Data"),
        (Tab2StrategyConfig, "Strategy", MockStrategyEngine()),
        (Tab3BacktestRunner, "Backtest", MockBacktester()),
        (Tab4ResultsAnalysis, "Results"),
        (Tab5ABTesting, "A/B Test", MockBacktester()),
        (Tab6LiveMonitoring, "Live"),
        (Tab7AdvancedAnalysis, "Research", MockAnalysisEngines()),
        (Tab8SystemSettings, "Settings"),
        (Tab9DataDownload, "Data Download"),
        (Tab10Help, "Help"),
    ]

    all_working = []
    all_broken = []

    for tab_info in tabs_to_test:
        tab_class = tab_info[0]
        tab_name = tab_info[1]
        tab_args = tab_info[2:] if len(tab_info) > 2 else []

        working, broken = test_tab_buttons(tab_class, tab_name, *tab_args)
        all_working.extend(working)
        all_broken.extend(broken)

    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    print(f"Total Working Buttons: {len(all_working)}")
    print(f"Total Broken Buttons: {len(all_broken)}")

    if all_working:
        print("\n✅ WORKING BUTTONS:")
        for button in all_working:
            print(f"  • {button}")

    if all_broken:
        print("\n❌ BROKEN BUTTONS:")
        for button, error in all_broken:
            print(f"  • {button} - {error}")

if __name__ == "__main__":
    main()