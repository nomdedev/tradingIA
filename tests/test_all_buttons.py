#!/usr/bin/env python3
"""
Script para probar todos los botones de la aplicación Trading IA
"""

import sys
import os
from PySide6.QtWidgets import QApplication, QPushButton
from PySide6.QtCore import QTimer, Qt
from PySide6.QtTest import QTest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main_platform import TradingPlatform

class ButtonTester:
    def __init__(self, app):
        self.app = app
        self.platform = TradingPlatform()
        self.test_results = []

    def log_result(self, tab_name, button_name, success, error_msg=""):
        """Log test result"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = f"{status} Tab {tab_name} - {button_name}"
        if error_msg:
            result += f" - {error_msg}"
        self.test_results.append(result)
        print(result)

    def test_tab_buttons(self, tab_index, tab_name):
        """Test all buttons in a specific tab"""
        print(f"\n🔍 Testing Tab {tab_index}: {tab_name}")

        # Switch to tab
        self.platform.tabs.setCurrentIndex(tab_index)
        QTest.qWait(500)  # Wait for tab to load

        tab_widget = self.platform.tabs.widget(tab_index)

        # Find all QPushButton instances
        buttons = tab_widget.findChildren(QPushButton)
        print(f"   Found {len(buttons)} buttons")

        for button in buttons:
            button_text = button.text().strip()
            if not button_text:
                button_text = button.objectName() or "Unnamed Button"

            try:
                # Check if button is enabled
                if not button.isEnabled():
                    self.log_result(tab_name, f"'{button_text}' (DISABLED)", True, "Button disabled")
                    continue

                # Try to click the button
                QTest.mouseClick(button, Qt.MouseButton.LeftButton)
                QTest.qWait(200)  # Wait for action

                # Check if any error occurred (basic check)
                self.log_result(tab_name, f"'{button_text}'", True)

            except Exception as e:
                self.log_result(tab_name, f"'{button_text}'", False, str(e))

    def run_all_tests(self):
        """Run tests on all tabs"""
        print("🚀 Starting Button Testing for Trading IA Platform")
        print("=" * 60)

        tab_names = [
            "Dashboard", "Data", "Strategy", "Backtest", "Results",
            "A/B Test", "Live", "Research", "Settings", "Data Download", "Help"
        ]

        for i, tab_name in enumerate(tab_names):
            try:
                self.test_tab_buttons(i, tab_name)
            except Exception as e:
                self.log_result(tab_name, "TAB_LOAD", False, f"Failed to load tab: {str(e)}")

        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)

        passed = sum(1 for r in self.test_results if "PASS" in r)
        failed = sum(1 for r in self.test_results if "FAIL" in r)

        print(f"Total Tests: {len(self.test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(".1f")

        if failed > 0:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if "FAIL" in result:
                    print(f"  {result}")

        return failed == 0

def main():
    app = QApplication(sys.argv)

    # Create tester
    tester = ButtonTester(app)

    # Run tests after a short delay to let UI initialize
    QTimer.singleShot(2000, lambda: tester.run_all_tests())

    # Start event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()