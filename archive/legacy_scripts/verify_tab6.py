import sys
import os
from PySide6.QtWidgets import QApplication
from src.gui.platform_gui_tab6_improved import Tab6LiveMonitor

# Mock Backtester
class MockBacktester:
    pass

def main():
    app = QApplication(sys.argv)
    
    # Create Tab 6
    tab6 = Tab6LiveMonitor(parent_platform=None, backtester_core=MockBacktester())
    tab6.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
