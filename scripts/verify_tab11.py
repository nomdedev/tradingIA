import sys
import os
from PySide6.QtWidgets import QApplication

# Add project root to path
sys.path.append(os.getcwd())

try:
    from src.gui.platform_gui_tab11_improved import Tab11RiskMetrics
    print("✅ Import successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

app = QApplication(sys.argv)

try:
    tab = Tab11RiskMetrics()
    print("✅ Instantiation successful")
except Exception as e:
    print(f"❌ Instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("✅ Tab 11 is ready")
