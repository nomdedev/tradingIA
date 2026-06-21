
import sys
import os
from PySide6.QtWidgets import QApplication, QLabel, QWidget

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.platform_gui_tab1_improved import Tab1DataManagement

class MockPlatform(QWidget):
    def __init__(self):
        super().__init__()
        self.data_manager = None

app = QApplication(sys.argv)
mock_platform = MockPlatform()
tab = Tab1DataManagement(mock_platform)

children = tab.findChildren(QLabel)
print(f"Found {len(children)} labels.")
for i, c in enumerate(children):
    style = c.styleSheet()
    text = c.text()
    print(f"Label {i}: Text='{text}', Style='{style[:20]}...'")
    if not style:
        print(f"  -> WARNING: Unstyled label! Parent: {c.parent()}")
