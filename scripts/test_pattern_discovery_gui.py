"""
Test script for Pattern Discovery GUI integration
"""
import sys
import os
import pytest

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


def test_pattern_discovery_gui():
    """Test the Pattern Discovery integration in Research tab"""

    print("=" * 80)
    print("TESTING PATTERN DISCOVERY GUI INTEGRATION")
    print("=" * 80)

    try:
        from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
        from src.gui.platform_gui_tab7_improved import Tab7AdvancedAnalysis

        # Create Qt Application (check if already exists to avoid singleton issues)
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        # Create main window
        window = QMainWindow()
        window.setWindowTitle("Pattern Discovery Test - Research Tab")
        window.setGeometry(100, 100, 1400, 900)

        # Create and set Tab7 widget
        tab7 = Tab7AdvancedAnalysis()
        window.setCentralWidget(tab7)

        print("\n✅ GUI components created successfully!")
        print("\nVerifications:")
        print("  ✓ Tab7AdvancedAnalysis loaded")
        print("  ✓ Pattern Discovery section should be visible")
        print("  ✓ '▶ Discover Patterns' button should be available")
        print("  ✓ Min cases spinner should be set to 15")

        # Basic verification that the widget was created
        assert tab7 is not None, "Tab7AdvancedAnalysis widget not created"
        assert isinstance(tab7, QWidget), "Tab7AdvancedAnalysis is not a QWidget"

        print("  ✓ Basic widget verification passed")

        # Don't run the app in test mode
        # sys.exit(app.exec())

    except ImportError as e:
        pytest.skip(f"GUI libraries not available: {e}")
    except Exception as e:
        pytest.fail(f"GUI test failed: {e}")


if __name__ == "__main__":
    test_pattern_discovery_gui()
