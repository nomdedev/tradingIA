import sys
import os
from PySide6.QtWidgets import QApplication, QWidget

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.gui.platform_gui_tab6_debug import Tab6LiveMonitor

class MockPlatform(QWidget):
    def __init__(self):
        super().__init__()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mock_platform = MockPlatform()
    
    print("Instantiating Tab6LiveMonitor...")
    print(f"Module: {Tab6LiveMonitor.__module__}")
    import inspect
    file_path = inspect.getfile(Tab6LiveMonitor)
    print(f"File: {file_path}")
    
    with open(file_path, 'r') as f:
        content = f.read()
        if "DEBUG" in content:
            print("DEBUG string FOUND in file content")
        else:
            print("DEBUG string NOT FOUND in file content")

    try:
        tab = Tab6LiveMonitor(mock_platform, None)
        print("Instantiation complete.")
        
        layout = tab.layout()
        print(f"Layout: {layout}")
        if layout:
            print(f"Layout count: {layout.count()}")
            
        children = tab.findChildren(QWidget)
        print(f"Children count: {len(children)}")
        
    except Exception as e:
        print(f"Error: {e}")
